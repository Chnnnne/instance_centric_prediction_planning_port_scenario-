import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers.polyline_net import PolylineNet
from .layers.plan_net import PlanNet
from .layers.fusion_net import FusionNet
from .layers.traj_decoder import TrajDecoder
from .layers.plan_decoder import PlanDecoder
from .layers.res_mlp import ResMLP
from .layers.score_decoder import ScoreDecoder
from .layers.score_decoder import get_transform,transform_to_ori
from common.math_utils import bernstein_poly, bezier_derivative, bezier_curve


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device
        self.agent_net = PolylineNet(  
            input_size=self.args.agent_input_size,
            hidden_size=self.args.agent_hidden_size,
            out_size=self.args.d_model)
        
        self.map_net = PolylineNet(
            input_size=self.args.map_input_size,
            hidden_size=self.args.map_hidden_size,
            out_size=self.args.d_model)
        
        self.rpe_net = nn.Sequential(
            nn.Linear(self.args.rpe_input_size, self.args.rpe_hidden_size),
            nn.LayerNorm(self.args.rpe_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_net = FusionNet(
            d_model=self.args.d_model, 
            d_edge=self.args.rpe_hidden_size, 
            dropout=self.args.dropout,
            update_edge=self.args.update_edge)
        
        self.refpath_encoder = nn.Sequential(
            ResMLP(20*2+20*2, self.args.decoder_hidden_size, self.args.decoder_hidden_size),
            nn.Linear(self.args.decoder_hidden_size, self.args.refpath_dim)
        )
        
        # self.plan_net = PlanNet(
        #     input_size=self.args.plan_input_size, 
        #     hidden_size=self.args.d_model
        # )
        
        self.traj_decoder = TrajDecoder(
            input_size=self.args.d_model,
            hidden_size=self.args.decoder_hidden_size, 
            n_order=self.args.bezier_order, 
            m=self.args.m,
            refpath_dim=self.args.refpath_dim)
        
        self.plan_decoder = PlanDecoder(
            input_size=self.args.d_model,
            hidden_size=self.args.decoder_hidden_size, 
            n_order=self.args.bezier_order, 
            m=self.args.m,
            refpath_dim=self.args.refpath_dim
        )
        if self.args.train_part == "back":
            self.scorer = None
        elif self.args.train_part == "front":
            pass
        else:# joint
            self.scorer = ScoreDecoder(
                n_order=self.args.bezier_order, 
            )
        
        # self.mat_T = self._get_T_matrix_bezier(n_order=self.args.bezier_order, n_step=50)
        # 会保存到model的state_dict中
        self.register_buffer('mat_T', self._get_T_matrix_bezier(n_order=self.args.bezier_order, n_step=50))
        
        self.relation_decoder = None
        if self.args.init_weights:
            self.apply(self._init_weights)

    def forward(self, batch_dict):
        agent_polylines, agent_polylines_mask = batch_dict['agent_feats'].cuda(), batch_dict['agent_mask'].cuda().bool() # [b,all_n-Max,20,13]  [b,all_n-Max,20]
        map_polylines, map_polylines_mask = batch_dict['map_feats'].cuda(), batch_dict['map_mask'].cuda().bool() # [b, map_element_num-Max, 20, 5]   [b, map_element_num-Max, 20]
        agent_feats = self.agent_net(agent_polylines, agent_polylines_mask) # B,N,20,13 + B,N,20 -> [B,N,d_agent]
        map_feats = self.map_net(map_polylines, map_polylines_mask) # B,N,20,5 + B,N,20 -> [b, map_element_num-Max,d_map]
        
        rpe, rpe_mask = batch_dict['rpe'].cuda(), batch_dict['rpe_mask'].cuda().bool()
        B, all_N, _, _= rpe.shape
        rpe_feats_valid = self.rpe_net(rpe[rpe_mask])  # (N, C)
        rpe_feats = rpe_feats_valid.new_zeros(B, all_N, all_N, rpe_feats_valid.shape[-1])
        rpe_feats[rpe_mask] = rpe_feats_valid
        
        agent_mask = (agent_polylines_mask.sum(dim=-1) > 0)  
        map_mask = (map_polylines_mask.sum(dim=-1) > 0)  
        agent_feats, map_feat = self.fusion_net(agent_feats, agent_mask, map_feats, map_mask, rpe_feats, rpe_mask) # agent_feats [b,all_n-Max, d_agent]
        
        # plan_traj, plan_traj_mask = batch_dict['plan_feat'].cuda(), batch_dict['plan_mask'].cuda().bool() # B,N,50,4   B,N,50
        # agent_feats, gate = self.plan_net(agent_feats, plan_traj, plan_traj_mask) # 
        
        candidate_refpaths_cords, candidate_refpaths_vecs, candidate_mask = batch_dict['candidate_refpaths_cords'].cuda(), batch_dict['candidate_refpaths_vecs'].cuda(), batch_dict['candidate_mask'].cuda().bool() # [b,all_n-Max,Max-N-Max, 2]   [b,all_n-Max,Max-N-Max]
        _, agent_N, M, _, _ = candidate_refpaths_cords.shape
        refpath_feats = self.refpath_encoder(torch.cat([candidate_refpaths_cords.reshape(B,agent_N,M,-1), candidate_refpaths_vecs.reshape(B,agent_N,M,-1)],dim=-1))# B,N,M,40 +40  ->B,N,M,64
        gt_refpath = batch_dict['gt_candts'].cuda() # [b, all_n-Max, Max-N-Max]
        gt_vel_mode = batch_dict['gt_vel_mode'].cuda() # [b, all_n-Max]
        cand_refpath_probs, param, traj_probs, param_with_gt,all_candidate_mask = self.traj_decoder(agent_feats, refpath_feats, gt_refpath, gt_vel_mode, candidate_mask)
        if torch.isnan(param).any():
            print("param contain nan", param)
        # 由贝塞尔控制点反推轨迹
        bezier_control_points = param.view(param.shape[0],
                                           param.shape[1],
                                           param.shape[2], -1, 2) # # B,N,3M,n_order*2 -> B, N, 3m, n_order+1, 2
        trajs = torch.matmul(self.mat_T, bezier_control_points) # B,N,3m,future_steps,2
        

        bezier_control_points_with_gt = param_with_gt.view(param_with_gt.shape[0],
                                                           param_with_gt.shape[1],
                                                           param_with_gt.shape[2], -1, 2) # B, N, 1, n_order+1*2 ->B, N, 1, n_order+1, 2
        traj_with_gt = torch.matmul(self.mat_T, bezier_control_points_with_gt) # B,N,1,future_steps,2


        ego_refpath_cords, ego_refpath_vecs, ego_vel_mode = batch_dict['ego_refpath_cords'].cuda(), batch_dict['ego_refpath_vecs'].cuda(), batch_dict['ego_vel_mode'].cuda()
        
        
        # 目标是输出自车的规划轨迹，我们需要给它提供自己的意图（也即一两个候选path）（根据5s点位于的坐标，来得到候选path）我们有了候选path，1.不用打分2.生成3条轨迹， 3.traj打分然后
        ego_refpath_feats = self.refpath_encoder(torch.cat([ego_refpath_cords.reshape(B,-1), ego_refpath_vecs.reshape(B, -1)],dim=-1)) # B,40+40 -> B, 64
        plan_params,  plan_param_with_gt = self.plan_decoder(agent_feats[:,0,:], ego_refpath_feats, ego_vel_mode)
        # plan_params, plan_traj_probs, plan_param_with_gt = self.plan_decoder(agent_feats[:,0,:], ego_refpath_feats, ego_vel_mode)
        if torch.isnan(plan_params).any():
            print("param contain nan", param)
        
        plan_bcp = plan_params.view(plan_params.shape[0], plan_params.shape[1], -1, 2) # B,3,(n_order+1)*2 -> B, 3, (n_order+1), 2
        plan_trajs = torch.matmul(self.mat_T, plan_bcp) # B,3,50,2

        plan_bcp_with_gt = plan_param_with_gt.view(plan_param_with_gt.shape[0], -1, 2) # B,(n_order+1)*2 -> B,(n_order+1),2 
        plan_traj_with_gt = torch.matmul(self.mat_T, plan_bcp_with_gt) # B, 50, 2
        if self.args.train_part == "front":
            scores, weights = None, None
        else: # back joint
            scores, weights = self.scorer(plan_trajs.clone(), plan_params.clone(), agent_feats[:,0,:], trajs.clone(), param.clone(), traj_probs, agent_polylines[:,:,-1,:], all_candidate_mask,agent_polylines_mask[:,:,-1], batch_dict['agent_vecs'].cuda(), batch_dict['agent_ctrs'].cuda(),self.mat_T)# B,3  B,8

        res = {"cand_refpath_probs": cand_refpath_probs, # B,N,M
                "traj_with_gt": traj_with_gt,#B,N,1,50,2
                "trajs": trajs, # B,N,3M,50,2
                "param":param, #B,N,3M,(n_order+1)*2
                "traj_probs": traj_probs, # B,N,3M
                "all_candidate_mask":all_candidate_mask, # B,N,3M
                "plan_trajs":plan_trajs, # B,3,50,2
                "plan_params":plan_params, # B,3,(n_order+1)*2
                # "plan_traj_probs":plan_traj_probs, # B,3
                "plan_traj_with_gt":plan_traj_with_gt, # B, 50, 2
                "scores":scores, # B,3
                "weights":weights
        }  
        # check_for_nan_in_dict(res)
        return res
        
    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.factorial(n_order) // (math.factorial(i) * math.factorial(n_order - i)) * (1.0 - ts)**(n_order - i) * ts**i
            # coeff = math.comb(n_order, i) * (1.0 - ts)**(n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                fan_in = m.embed_dim
                fan_out = m.embed_dim
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                nn.init.uniform_(m.in_proj_weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(m.q_proj_weight)
                nn.init.xavier_uniform_(m.k_proj_weight)
                nn.init.xavier_uniform_(m.v_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
            if m.bias_k is not None:
                nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
            if m.bias_v is not None:
                nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
        
    def compute_model_metrics(self, input_dict, output_dict):
        '''
        对于ego和agent，计算的指标都是多模态轨迹中，选取了一条（根据prob或者score）计算的ade、fde、mr、jerk ，也称为 min ade、fde、mr、jerk
        对于打分模块，由于是分阶段训练，因此front阶段计算的ego的指标，都是三条平均的ade、fde、jerk
        '''
        pred_mask = (input_dict['candidate_mask'].sum(dim=-1) > 0).cuda() # B,N,M-> B,N
        # 计算目标点有关的指标
        target_top = 0
        gt_prob = input_dict['gt_candts'].cuda()[pred_mask]# B,N,M -> S,M
        if gt_prob.dim() == 3:
            gt_prob = gt_prob.squeeze(dim=-1)
        valid_batch_size, label_num = gt_prob.size() # S, M
        gt_label = torch.argmax(gt_prob, -1) # S,获取one-hot的idx
        # target_prob = output_dict['target_probs'][pred_mask]
        target_prob = output_dict['cand_refpath_probs'][pred_mask] # B,N,M -> S,M
        
        max_probs, max_indices = torch.max(target_prob, dim=1) # S
        # 计算target的准确率
        for i in range(valid_batch_size):
            if gt_label[i] ==max_indices[i]:
                target_top += 1
        '''   
        target_gt = input_dict["gt_preds"].cuda()[pred_mask][:, -1, :2]# B, N, 50, 2 -> S, 50, 2 -> S, 2
        target_candidate = input_dict["tar_candidate"].cuda()[pred_mask] # B, N, M, 2 -> S, M, 2
        offset = output_dict['pred_offsets'][pred_mask] # B,N,M,2 -> S,M,2
        target_pred = target_candidate[torch.arange(batch_size), max_indices] + offset[torch.arange(batch_size), max_indices]# S,2
        # endpoint_pred = output_dict['trajs'].cuda()[pred_mask][:,:,-1,:2] #  B,N,m,future_steps,2 -> S,m,future_steps,2 -> S,M,2
        # endpoint_pred = endpoint_pred[torch.arange(batch_size), max_indices] # S,2

        # 计算target的FDE
        rmse = torch.linalg.norm(target_pred - target_gt, dim=-1)# S,
        target_fde = torch.sum(rmse)
        '''
        # 计算轨迹有关的指标
        score = output_dict['traj_probs'][pred_mask] # B, N, 3M -> S, 3M
        traj_max_probs, traj_max_indices = torch.max(score, dim=1) # S
        trajs = output_dict['trajs'][pred_mask] # B, N, 3M, 50, 2 -> S, 3M, 50, 2
        traj_pred = trajs[torch.arange(valid_batch_size), traj_max_indices] # S, 50, 2 得到得分最高的轨迹和真值计算指标
        # traj_pred = trajs[torch.arange(batch_size), traj_max_indices].view(batch_size, 50, 2) 
        traj_gt = input_dict["gt_preds"].cuda()[pred_mask]# B N 50 2 -> S 50 2
        
        # 计算traj的ADE
        squared_distance = torch.sum((traj_pred - traj_gt) ** 2, dim=2) # S,50
        distance = torch.sqrt(squared_distance)# S,50
        traj_ade = torch.mean(distance, dim=1).sum() # S -> 1
       
        # 计算tarj的FDE
        fde =  torch.linalg.norm(traj_pred[:, -1, :] - traj_gt[:, -1, :], dim=-1) # S,2 - S,2    norm -> S
        traj_fde = torch.sum(fde)# S->1

        # 计算MR
        missing_num = torch.sum(fde > 3) # 1
        
        # 计算Jerk
        param = output_dict['param'][pred_mask][torch.arange(valid_batch_size), traj_max_indices].reshape(valid_batch_size,self.args.bezier_order+1,2) # #B,N,3M,(n_order+1)*2 ->S,3M,(n_order+1)*2 -> S,(n_order+1),2
        vec_param = bezier_derivative(param) # S,n_order,2
        acc_param = bezier_derivative(vec_param) # S, n_order-1, 2
        jerk_param = bezier_derivative(acc_param) # S, n_order-2, 2
        t_values = torch.linspace(0,1,20).cuda()
        jerk_vector = bezier_curve(jerk_param,t_values)/5 # S,20,2
        jerk_scaler = torch.linalg.norm(jerk_vector,dim=-1) # S,20
        RMS_jerk = torch.sqrt(torch.mean(jerk_scaler**2, dim=-1)) # S 
        RMS_jerk = torch.sum(RMS_jerk)# S->1

        # 计算自车指标
        # ADE、FDE、MR、Jerk
        ego_gt_traj = input_dict['ego_gt_traj'].cuda() # B,50,2
        batch_size,_,_ = ego_gt_traj.shape
        plan_trajs = output_dict['plan_trajs'] # B,3,50,2
        plan_param = output_dict['plan_params'].reshape(batch_size, 3, self.args.bezier_order+1, 2) # B,3,(n+1),2

        if output_dict['scores'] != None:
            # 计算min ade fde ...
            scores = output_dict['scores'] # B,3
            score_max_indices = torch.argmax(scores, dim=-1) # B
            plan_traj_best = plan_trajs[torch.arange(batch_size), score_max_indices] # B,50,2
            # ade
            plan_ade = torch.sum(torch.mean(torch.linalg.norm(plan_traj_best - ego_gt_traj,dim=-1),dim=-1)) # B,50,2 -> B,50->B->1
            # fde
            plan_fde = torch.linalg.norm(plan_traj_best[:,-1,:] - ego_gt_traj[:,-1,:], -1)# B,2 -> B
            plan_fde_sum = torch.sum(plan_fde) # B -> 1
            # mr
            plan_missing_num = torch.sum(plan_fde>3)
            # jerk
            plan_param = plan_param[torch.arange(batch_size), score_max_indices]# B,3,(n_order+1),2 -> B,n_order+1,2
            plan_vec_param = bezier_derivative(plan_param) # B,n_order,2
            plan_acc_param = bezier_derivative(plan_vec_param) # B, n_order-1, 2
            plan_jerk_param = bezier_derivative(plan_acc_param) # B, n_order-2, 2
            plan_jerk_vector = bezier_curve(plan_jerk_param,t_values)/5 # B,20,2
            plan_jerk_scaler = torch.linalg.norm(plan_jerk_vector,dim=-1) # B,20
            plan_RMS_jerk = torch.sqrt(torch.mean(plan_jerk_scaler**2, dim=-1)) # B
            plan_RMS_jerk = torch.sum(RMS_jerk)# B->1
        else:
            # 计算 mean ade fde ...
            # mean ade
            plan_ade = torch.sum(torch.mean(torch.mean(torch.linalg.norm(plan_trajs - ego_gt_traj[:,None,:,:],dim=-1),dim=-1))) # B,3,50,2 -> B,3,50 -> B,3 -> B ->1
            # mean fde
            plan_fde = torch.linalg.norm(plan_trajs[:,:,-1,:] - ego_gt_traj[:,None,-1,:], -1)# B,3,2 - B,1,2 -> B,3
            plan_fde_sum = plan_fde.mean(-1).sum() # B,3->B->1
            # mr
            plan_missing_num = batch_size - (plan_fde < 3).any(dim=-1).sum() # B,3 -> B -> 1
            # jerk
            plan_vec_param = bezier_derivative(plan_param) # B,3,n_order,2
            plan_acc_param = bezier_derivative(plan_vec_param) # B,3, n_order-1, 2
            plan_jerk_param = bezier_derivative(plan_acc_param) # B,3, n_order-2, 2
            plan_jerk_vector = bezier_curve(plan_jerk_param,t_values)/5 # B,3,20,2
            plan_jerk_scaler = torch.linalg.norm(plan_jerk_vector,dim=-1) # B,3,20
            plan_RMS_jerk = torch.sqrt(torch.mean(plan_jerk_scaler**2, dim=-1)) # B,3
            plan_RMS_jerk = torch.sum(torch.mean(RMS_jerk,dim=-1))# B,3->B->1



             
        


        return valid_batch_size, target_top, traj_ade, traj_fde, missing_num, RMS_jerk, batch_size, plan_ade, plan_fde_sum, plan_missing_num, plan_RMS_jerk
    
# 函数检查dict中是否存在包含NaN的tensor
def check_for_nan_in_dict(data_dict):
    for key, value in data_dict.items():
        if torch.isnan(value).any():
            print(f"NaN found in tensor associated with key '{key}':\n{value}")