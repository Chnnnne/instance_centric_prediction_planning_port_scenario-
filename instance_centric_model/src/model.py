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
        
        
        candidate_refpaths_cords, candidate_refpaths_vecs, candidate_mask = batch_dict['candidate_refpaths_cords'].cuda(), batch_dict['candidate_refpaths_vecs'].cuda(), batch_dict['candidate_mask'].cuda().bool() # [b,N,M, 20,2]   [b,N,M]
        _, agent_N, M, _, _ = candidate_refpaths_cords.shape
        refpath_feats = self.refpath_encoder(torch.cat([candidate_refpaths_cords.reshape(B,agent_N,M,-1), candidate_refpaths_vecs.reshape(B,agent_N,M,-1)],dim=-1))# B,N,M,40 +40  ->B,N,M,64
        gt_refpath = batch_dict['gt_candts'].cuda() # [b, all_n-Max, Max-N-Max]
        gt_vel_mode = batch_dict['gt_vel_mode'].cuda() # [b, all_n-Max]
        # agent traj decoder
        cand_refpath_probs, param, traj_probs, param_with_gt,all_candidate_mask = self.traj_decoder(agent_feats, refpath_feats, gt_refpath, gt_vel_mode, candidate_mask)
        if torch.isnan(param).any():
            print("param contain nan", param)
        bezier_control_points = param.view(param.shape[0],
                                           param.shape[1],
                                           param.shape[2], -1, 2) # # B,N,3M,n_order*2 -> B, N, 3m, n_order+1, 2
        trajs = torch.matmul(self.mat_T, bezier_control_points) # B,N,3m,future_steps,2
        
        bezier_control_points_with_gt = param_with_gt.view(param_with_gt.shape[0],
                                                           param_with_gt.shape[1],
                                                           param_with_gt.shape[2], -1, 2) # B, N, 1, n_order+1*2 ->B, N, 1, n_order+1, 2
        traj_with_gt = torch.matmul(self.mat_T, bezier_control_points_with_gt) # B,N,1,future_steps,2


        #B,M,20,2      B,M,20,2        B       B,M      B,M
        ego_refpath_cords, ego_refpath_vecs, ego_vel_mode, ego_cand_mask, ego_gt_cand = batch_dict['ego_refpath_cords'].cuda(), batch_dict['ego_refpath_vecs'].cuda(), batch_dict['ego_vel_mode'].cuda(), batch_dict['ego_cand_mask'].cuda().bool(), batch_dict['ego_gt_cand'].cuda()
        ref_M = ego_refpath_cords.shape[1]
        
        ego_refpath_feats = self.refpath_encoder(torch.cat([ego_refpath_cords.reshape(B,ref_M,-1), ego_refpath_vecs.reshape(B,ref_M, -1)],dim=-1)) # B,M,20*2 + 20*2  -> B, M, 64
        # ego traj decoder
        plan_cand_refpath_probs, plan_params, plan_traj_probs, plan_param_with_gt, plan_all_candidate_mask = self.plan_decoder(agent_feats[:,0,:], ego_refpath_feats, ego_vel_mode, ego_cand_mask, ego_gt_cand)

        if torch.isnan(plan_params).any():
            print("param contain nan", param)
        
        plan_bcp = plan_params.view(plan_params.shape[0], plan_params.shape[1], -1, 2) # B,3M,(n_order+1)*2 -> B, 3M, (n_order+1), 2
        plan_trajs = torch.matmul(self.mat_T, plan_bcp) # B,3M,50,2

        plan_bcp_with_gt = plan_param_with_gt.view(plan_param_with_gt.shape[0], -1, 2) # B,(n_order+1)*2 -> B,(n_order+1),2 
        plan_traj_with_gt = torch.matmul(self.mat_T, plan_bcp_with_gt) # B, 50, 2
        
        if self.args.train_part == "front":
            scores, weights = None, None
        else: # back joint
            scores, weights = self.scorer(plan_trajs.clone(), plan_params.clone(), agent_feats[:,0,:], trajs.clone(), param.clone(), traj_probs, agent_polylines[:,:,-1,:], all_candidate_mask,plan_all_candidate_mask, agent_polylines_mask[:,:,-1], batch_dict['agent_vecs'].cuda(), batch_dict['agent_ctrs'].cuda(),self.mat_T)# B,3  B,8

        res = {"cand_refpath_probs": cand_refpath_probs, # B,N,M
                "trajs": trajs, # B,N,3M,50,2
                "param":param, #B,N,3M,(n_order+1)*2
                "traj_probs": traj_probs, # B,N,3M
                "traj_with_gt": traj_with_gt,#B,N,1,50,2
                "all_candidate_mask":all_candidate_mask, # B,N,3M
                
                "plan_cand_refpath_probs":plan_cand_refpath_probs, # B,M
                "plan_trajs":plan_trajs, # B,3M,50,2
                "plan_params":plan_params, # B,3M,(n_order+1)*2
                "plan_traj_probs":plan_traj_probs, # B,3M
                "plan_traj_with_gt":plan_traj_with_gt, # B, 50, 2
                "plan_all_candidate_mask":plan_all_candidate_mask,# B,3M
                "scores":scores, # B,3M
                "weights":weights # B,8
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
        #######################################################################################
        # 预测指标
        #######################################################################################
        T, D = 50, 2
        M = output_dict['param'].shape[2]
        plan_M = output_dict['plan_params'].shape[1]
        batch_size, N = pred_mask.shape
        order = self.args.bezier_order + 1
        t_values = torch.linspace(0,1,50).cuda()
        k=3

        target_top = 0
        gt_prob = input_dict['gt_candts'].cuda()[pred_mask]# B,N,M -> S,M
        if gt_prob.dim() == 3:
            gt_prob = gt_prob.squeeze(dim=-1)
        valid_batch_size, label_num = gt_prob.size() # S, M
        gt_label = torch.argmax(gt_prob, -1) # S,获取one-hot的idx
        target_prob = output_dict['cand_refpath_probs'][pred_mask] # B,N,M -> S,M
        
        max_probs, max_indices = torch.max(target_prob, dim=1) # S
        # 计算target的准确率
        for i in range(valid_batch_size):
            if gt_label[i] ==max_indices[i]:
                target_top += 1


        param = output_dict['param'][pred_mask].reshape(valid_batch_size, M,order, 2) # B,N,3M,(n+1)*2,    S,3M,(n+1)*2 =>  S,3M,(n+1), 2 
        traj_probs = output_dict['traj_probs'][pred_mask] # B, N, 3M -> S, 3M
        trajs = output_dict['trajs'][pred_mask] # B, N, 3M, 50, 2 -> S, 3M, 50, 2
        traj_gt = input_dict["gt_preds"].cuda()[pred_mask].unsqueeze(1) # B N 50 2 -> S,1,50,2


        traj_topk_probs, traj_topk_indices = traj_probs.topk(k=3, dim=-1) # S,K
        topk_trajs = trajs.gather(1, traj_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,D))# S,K,50,2
        topk_param = param.gather(1, traj_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,order, 2))# S,K,(n+1),2

        traj_top1_probs, traj_top1_indices = traj_probs.topk(k=1, dim=-1) #S,1
        top1_trajs = trajs.gather(1, traj_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,D))# S,1,50,2
        top1_param = param.gather(1, traj_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,order, 2))# S,1,(n+1),2
        
        _, mink_ade = get_minK_ade(topk_trajs, traj_gt) # 1
        _, min1_ade = get_minK_ade(top1_trajs, traj_gt) # 1
        mink_fde_part, mink_fde = get_minK_fde(topk_trajs, traj_gt)
        min1_fde_part, min1_fde = get_minK_fde(top1_trajs, traj_gt)
        mink_brier_fde, brier_score = get_minK_brier_FDE(topk_trajs, traj_gt, traj_topk_probs)

        mink_mr = get_minK_mr(mink_fde_part)
        min1_mr = get_minK_mr(min1_fde_part)
        mink_RMS_jerk = get_minK_jerk(topk_param,t_values)
        min1_RMS_jerk = get_minK_jerk(top1_param,t_values)

        # 计算curvature、lateral

        # 计算pred_agent_yaw、 agent_gt_yaw
        mink_ahe = get_minK_ahe(topk_trajs, traj_gt) # S,K,50,2    S,1,50,2    -> 1
        min1_ahe = get_minK_ahe(top1_trajs, traj_gt)
        mink_fhe = get_minK_fhe(topk_trajs, traj_gt)
        min1_fhe = get_minK_fhe(top1_trajs, traj_gt)
        
        #######################################################################################
        # 规划指标
        #######################################################################################

        ego_gt_traj = input_dict['ego_gt_traj'].unsqueeze(1).cuda() # B,1,50,2
        plan_trajs = output_dict['plan_trajs'] # B,3M,50,2
        plan_param = output_dict['plan_params'].reshape(batch_size, plan_M, self.args.bezier_order+1, 2) # B,3M,(n+1),2
        
        if output_dict['scores'] != None:
            plan_scores = output_dict['scores']
            plan_scores = score_to_prob(plan_scores)
            # plan_cost = output_dict['scores'] # B,3M
            # min_val = plan_cost.min(dim=-1).values.unsqueeze(-1) # B
            # max_val = plan_cost.max(dim=-1).values.unsqueeze(-1) # B
            # plan_cost = (plan_cost - min_val)/(max_val - min_val) # B,3M / B,1
            # plan_supl = output_dict['plan_traj_probs'] # B,3M
            # plan_scores = plan_supl * plan_cost #B,3M

        elif "plan_traj_probs" in output_dict:
            plan_scores = output_dict['plan_traj_probs'] # B,3M
            
        plan_topk_scores, score_topk_indices = torch.topk(plan_scores, k=k, dim=-1) # B,3M -> B,K   B,K
        plan_topk_traj = plan_trajs.gather(1, score_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,D)) # B,3M,50,2 -> B,K,50,2
        plan_topk_param = plan_param.gather(1, score_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,order, 2))# B,3M,n+1,2->B,K,n+1,2
        
        plan_top1_scores, score_top1_indices = torch.topk(plan_scores, k=1, dim=-1) # B,1   B,1  
        plan_top1_traj = plan_trajs.gather(1, score_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,D)) # B,3M,50,2 ->  B,1,50,2
        plan_top1_param = plan_param.gather(1, score_top1_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,order, 2))# ->B,1,n+1,2
        #ade
        plan_mink_ade_part, plan_mink_ade = get_minK_ade(plan_topk_traj, ego_gt_traj) # B,   1
        plan_min1_ade_part, plan_min1_ade = get_minK_ade(plan_top1_traj, ego_gt_traj)
        #fde
        plan_mink_fde_part, plan_mink_fde = get_minK_fde(plan_topk_traj, ego_gt_traj)
        plan_min1_fde_part, plan_min1_fde = get_minK_fde(plan_top1_traj, ego_gt_traj)
        # brier-fde
        plan_mink_brier_fde, plan_brier_score = get_minK_brier_FDE(plan_topk_traj, ego_gt_traj, plan_topk_scores)
        # mr
        plan_mink_mr = get_minK_mr(plan_mink_fde_part)
        plan_min1_mr = get_minK_mr(plan_min1_fde_part)
        # jerk
        plan_mink_RMS_jerk = get_minK_jerk(plan_topk_param,t_values) 
        plan_min1_RMS_jerk = get_minK_jerk(plan_top1_param,t_values) 
        # collision 指标
        agent_vecs = input_dict['agent_vecs'].cuda() # B,N,2
        agent_ctrs = input_dict['agent_ctrs'].cuda() # B,N,2
        transforms = get_transform(agent_vecs) # B,N,2 -> B,N,2,2
        plan_top1_traj_transform = transform_to_ori(plan_top1_traj, transforms[:,0,:,:], agent_ctrs[:,0,:]).squeeze(1) # B,1,50,2 + B,2,2 + B,2-> B,50,2
        top1_trajs_transofrm = transform_to_ori(top1_trajs, transforms[pred_mask], agent_ctrs[pred_mask]).squeeze(1) # S,1,50,2  + S,2,2 + S,2 -> S,50,2

        plan_top1_traj_transform_mask = plan_top1_traj_transform.unsqueeze(1).repeat(1,N,1,1)[pred_mask] # B,50,2 -> B,N,50,2 -> S,50,2
        agent_min_dist = torch.linalg.norm(plan_top1_traj_transform_mask - top1_trajs_transofrm, dim=-1).min(dim=-1).values # S,50,2 -> S,50->S
        risk_num = torch.sum(agent_min_dist < 8)
        agent_dist = agent_min_dist.sum(-1)
        
        # curvature/lateral
        plan_vec_param = bezier_derivative(plan_top1_param) # B,1,n+1,2 -> B,1,n,2
        plan_vec_vectors = bezier_curve(plan_vec_param, t_values)/5 # B,1,n_order,2 -> B,1,50,2
        plan_vec_scaler = torch.linalg.norm(plan_vec_vectors, dim=-1) # B,1,50

        plan_acc_param = bezier_derivative(plan_vec_param) # B,1,n-1, 2
        plan_acc_vectors = bezier_curve(plan_acc_param, t_values)/25 # B, 1,n_order-1,2 -> B,1,50,2

        plan_vx, plan_vy = plan_vec_vectors[...,0], plan_vec_vectors[...,1]# B,1,50
        plan_ax, plan_ay = plan_acc_vectors[...,0], plan_acc_vectors[...,1]
        epsilon = 1e-4
        plan_curvature = torch.abs(plan_vx * plan_ay - plan_vy * plan_ax)/((plan_vx**2 + plan_vy**2+epsilon)**1.5) #B,1,50
        plan_lateral_acceleration = (plan_vec_scaler**2 * plan_curvature).mean(-1).min(dim=-1).values.sum(-1)
        plan_curvature = plan_curvature.mean(-1).min(dim=-1).values.sum()# 1

        # ahe/fhe
        plan_min1_ahe = get_minK_ahe(plan_top1_traj,ego_gt_traj)
        plan_min1_fhe = get_minK_fhe(plan_top1_traj,ego_gt_traj)


        '''
        else:
            # 计算 mean ade fde ...
            # mean ade
            plan_ade = torch.sum(torch.mean(torch.mean(torch.linalg.norm(plan_trajs - ego_gt_traj[:,None,:,:],dim=-1),dim=-1))) # B,3,50,2 -> B,3,50 -> B,3 -> B ->1
            # mean fde
            plan_fde = torch.linalg.norm(plan_trajs[:,:,-1,:] - ego_gt_traj[:,None,-1,:], dim=-1)# B,3,2 - B,1,2 -> B,3
            plan_fde_sum = plan_fde.mean(-1).sum() # B,3->B->1
            # mr
            plan_missing_num = batch_size - (plan_fde < 3).any(dim=-1).sum() # B,3 -> B -> 1
            # jerk
            plan_vec_param = bezier_derivative(plan_param) # B,3,n_order,2
            plan_acc_param = bezier_derivative(plan_vec_param) # B,3, n_order-1, 2
            plan_jerk_param = bezier_derivative(plan_acc_param) # B,3, n_order-2, 2
            plan_jerk_vector = bezier_curve(plan_jerk_param,t_values)/(5**3) # B,3,20,2
            plan_jerk_scaler = torch.linalg.norm(plan_jerk_vector,dim=-1) # B,3,20
            plan_RMS_jerk = torch.sqrt(torch.mean(plan_jerk_scaler**2, dim=-1)) # B,3
            plan_RMS_jerk = torch.sum(torch.mean(RMS_jerk,dim=-1))# B,3->B->1
        '''

        metric_dict ={"valid_batch_size":(valid_batch_size, 0), "batch_size":(batch_size, 0), 
                      "target_top":(target_top, 1), 
                      "mink_ade":(mink_ade,1), "min1_ade":(min1_ade,1), 
                      "mink_fde":(mink_fde,1), "min1_fde":(min1_fde,1), "mink_brier_fde":(mink_brier_fde, 1),  "brier_score":(brier_score, 1),
                      "mink_mr":(mink_mr,1), "min1_mr":(min1_mr,1), 
                      "mink_RMS_jerk": (mink_RMS_jerk,1), "min1_RMS_jerk":(min1_RMS_jerk,1), 
                      "mink_ahe":(mink_ahe,1), "min1_ahe": (min1_ahe,1),
                      
                      "plan_mink_ade":(plan_mink_ade,2), "plan_min1_ade":(plan_min1_ade,2), 
                      "plan_mink_fde":(plan_mink_fde,2), "plan_min1_fde":(plan_min1_fde,2), "plan_mink_brier_fde":(plan_mink_brier_fde,2), "plan_brier_score":(plan_brier_score, 2),
                      "plan_mink_mr":(plan_mink_mr, 2),"plan_min1_mr":(plan_min1_mr,2),
                      "plan_mink_RMS_jerk":(plan_mink_RMS_jerk,2),"plan_min1_RMS_jerk":(plan_min1_RMS_jerk,2), 
                      "risk_num":(risk_num,2),"agent_dist":(agent_dist,2),
                      "plan_curvature":(plan_curvature,2),"plan_lateral_acceleration":(plan_lateral_acceleration,2),
                      "plan_min1_ahe":(plan_min1_ahe,2),"plan_min1_fhe":(plan_min1_fhe,2)}
        return metric_dict


def score_to_prob(scores):
    # scores B,3M
    # min_val = scores.min(dim = -1,keepdim=True).values # B,1
    # max_val = scores.max(dim = -1,keepdim=True).values # B,1
    # scores = (scores - min_val)/(max_val - min_val) # 归一化
    # scores = scores**2
    scores = F.softmax(scores, dim=-1)
    return scores

def get_yaw(traj):
    '''
    traj:B,N,T,2 B个agent N条轨迹
    '''
    vec_vector = torch.diff(traj, dim=-2) #  B,N,T-1,2
    yaw = torch.atan2(vec_vector[...,1],vec_vector[...,0]) # B,N,T-1
    return yaw

def get_minK_ahe(proposed_traj,gt_traj):
    '''
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return 
        - ahe B,W,49 - B,1,49  -> B,W,49 -> B,W -> B -> sum
    '''  
    traj_yaw = get_yaw(proposed_traj) # B,W,49
    gt_yaw = get_yaw(gt_traj) # B,1,49
    yaw_diff = torch.abs(principal_value(traj_yaw - gt_yaw)).mean(-1).min(dim=-1).values.sum(-1) 
    return yaw_diff

def get_minK_fhe(proposed_traj, gt_traj):
    '''
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return:
        - afe:B    B,W,49 - B,1,49  -> B,W,49, -> B,W -> B -> sum

    '''  
    traj_yaw = get_yaw(proposed_traj) # B,W,49
    gt_yaw = get_yaw(gt_traj) # B,1,49
    yaw_diff = torch.abs(principal_value(traj_yaw - gt_yaw))[...,-1].min(dim=-1).values.sum(-1) 
    return yaw_diff

def get_minK_ade(proposed_traj,gt_traj):
    '''
    求预测的轨迹（每个agent有W条）和真值轨迹（1条）的最小ade
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return:
        - ade_part:B    B,W,50,2 - B,1,50,2 -> B,W,50 -> B,W -> B, 
        - ade:1    B->1 
    '''
    ade_part = torch.linalg.norm(proposed_traj - gt_traj,dim=-1).mean(-1).min(dim=-1).values
    ade = ade_part.sum(-1)
    return ade_part,ade

def get_minK_fde(proposed_traj, gt_traj):
    '''
    求预测的轨迹（每个agent有W条）和真值轨迹（1条）的最小fde
    input:
        - proposed_traj B,W不定,50,2
        - gt_traj B,1,50,2
    return:
        - fde_part:B    B,W,2 - B,1,2 -> B,W -> B
        - fde:1       B->1
    '''
    fde_part = torch.linalg.norm(proposed_traj[:,:,-1,:] - gt_traj[:,:,-1,:], dim=-1).min(dim=-1).values
    fde = fde_part.sum(-1)
    return fde_part, fde


def distance_metric(traj_candidate: torch.Tensor, traj_gt: torch.Tensor):
    """
    input:
    - S,m, 100
    - S,100

    return:
    S, M
    compute the distance between the candidate trajectories and gt trajectory
    :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
    :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
    :return: distance, torch.Tensor, [batch_size, M] or [1, M]
    """
    assert traj_gt.dim() == 2, "Error dimension in ground truth trajectory"
    if traj_candidate.dim() == 3:
        # batch case
        pass

    elif traj_candidate.dim() == 2:
        traj_candidate = traj_candidate.unsqueeze(1)
    else:
        raise NotImplementedError

    assert traj_candidate.size()[2] == traj_gt.size()[1], "Miss match in prediction horizon!"

    _, M, horizon_2_times = traj_candidate.size()
    dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times / 2), 2)
    # S,m,100 - S,1,100       (S,m,100)**2        view (S,m,50,2)
    dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)# (S,m,50,2)---sum-->(S,m,50)  --max-> S,m
    dis = torch.sqrt(dis)
    # dis= torch.sqrt(torch.sum(dis, dim=3)) # (S,m,50)  
    # weight = torch.linspace(0.5,1.5,int(horizon_2_times / 2)).cuda()
    # res = torch.sum(dis*weight, axis = 2)
    return dis

def get_minK_brier_FDE(proposed_traj, gt_traj, pred_prob):
    '''
    - proposed_traj: B,W不定,50,2
    - gt_traj: B,1,50,2
    - pred_prob: B,W
    '''
    B,W,T,D = proposed_traj.shape
    # 计算 FDE
    distances = torch.linalg.norm(proposed_traj[:,:,-1,:] - gt_traj[:,:,-1,:], dim=-1)#B,W

    dist = distance_metric(proposed_traj.view(B,W,-1), gt_traj.reshape(B,-1)) # B,W,100 + B,100 -> B,W
    gt_prob = F.softmax(dist,dim=-1) # B,W
    # 计算 Brier Score（示例，这里假设真实的概率分布是均匀的）

    brier_score = torch.sum((pred_prob - gt_prob) ** 2)# B,W 
    
    # 加权 FDE
    weighted_fde = distances * gt_prob # B,W 
    
    return torch.min(weighted_fde,dim=-1).values.sum(-1), brier_score

def get_minK_mr(fde_part, dis=3):
    '''
    求预测轨迹（每个agent有W条）和真值轨迹（每个agent1条）的最小fde是否大于dis（）
    input: fde_part : B
    '''
    return (fde_part > dis).sum(-1)

def get_minK_jerk(param, t_values):
    '''
    param: [B,W,n+1, 2]
    '''
    plan_vec_param = bezier_derivative(param) # B,W, n_order,2
    plan_acc_param = bezier_derivative(plan_vec_param) # B, W,n_order-1, 2
    plan_jerk_param = bezier_derivative(plan_acc_param) # B,W, n_order-2, 2
    plan_jerk_vector = bezier_curve(plan_jerk_param,t_values)/(5**3) # B,W, 20,2
    plan_jerk_scaler = torch.linalg.norm(plan_jerk_vector,dim=-1) # B,W, 20
    plan_RMS_jerk = torch.sqrt(torch.mean(plan_jerk_scaler**2, dim=-1)).min(dim=-1).values # B,W,20 -均方根-> B,W->B
    plan_RMS_jerk = torch.sum(plan_RMS_jerk)# B->1
    return plan_RMS_jerk

def get_transform(agent_vecs):
    # B,N,2
    # B,N,2,2
    B, N, _ = agent_vecs.shape
    cos_, sin_ = agent_vecs[:,:,0], agent_vecs[:,:,1] # B,N
    rot_matrix = torch.zeros(B, N, 2, 2, device=agent_vecs.device)
    rot_matrix[:,:,0,0] = sin_
    rot_matrix[:,:,0,1] = cos_
    rot_matrix[:,:,1,0] = -cos_
    rot_matrix[:,:,1,1] = sin_
    # one = torch.cat([sin_,cos_], dim=-1).unsqueeze(-2) # B,N,1,2
    # two = torch.cat([-cos_, sin_], dim= -1).unsqueeze(-2) # B,N,1,2
    return rot_matrix


def transform_to_ori(origin_feat, transforms, ctrs, mask = None):
    '''
    feat: B,N,M,50,2 or B,3,50,2 or B,1,50,2
    transforms: B,N,2,2 or B,2,2
    ctrs: B,N,2
    mask : B,N,3m,
    '''
    feat = origin_feat.clone()
    squeeze_one_flag = False
    if len(feat.shape) == 4:
        squeeze_one_flag = True
        feat = feat.unsqueeze(1) # B,1,3,50,2
        transforms = transforms.unsqueeze(1) # B,1,2,2
        ctrs = ctrs.unsqueeze(1) # B,1,2
    # B,N,M,50,2
    rot_inv = transforms.transpose(-2, -1) 
    feat[..., 0:2] = torch.matmul(feat[..., 0:2], rot_inv[:,:,None,:,:]) + ctrs[:,:,None,None,:] # B,N,M,50,2@B,N,1,2,2 + B,N,1,1,2
    if squeeze_one_flag:
        feat = feat.squeeze(1)
    if mask != None:
        B,N,M,T,D = feat.shape
        mask = (~mask.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,T,D))
        feat = feat.masked_fill(mask,0.0)
    return feat


# 函数检查dict中是否存在包含NaN的tensor
def check_for_nan_in_dict(data_dict):
    for key, value in data_dict.items():
        if torch.isnan(value).any():
            print(f"NaN found in tensor associated with key '{key}':\n{value}")

def principal_value(angle, min_= -math.pi):
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    S,N,49
    """
    assert torch.all(torch.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * math.pi) + min_

    return lhs