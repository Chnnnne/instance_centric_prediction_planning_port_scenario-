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
        
        self.plan_net = PlanNet(
            input_size=self.args.plan_input_size, 
            hidden_size=self.args.d_model
        )
        
        self.traj_decoder = TrajDecoder(
            input_size=self.args.d_model,
            hidden_size=self.args.decoder_hidden_size, 
            n_order=self.args.bezier_order, 
            m=self.args.m)
        
        # self.mat_T = self._get_T_matrix_bezier(n_order=self.args.bezier_order, n_step=50)
        # 会保存到model的state_dict中
        self.register_buffer('mat_T', self._get_T_matrix_bezier(n_order=self.args.bezier_order, n_step=50))
        
        self.relation_decoder = None
        if self.args.init_weights:
            self.apply(self._init_weights)
        
    def forward(self, batch_dict):
        agent_polylines, agent_polylines_mask = batch_dict['agent_feats'].cuda(), batch_dict['agent_mask'].cuda().bool() # [b,all_n-Max,20,13]  [b,all_n-Max,20]
        map_polylines, map_polylines_mask = batch_dict['map_feats'].cuda(), batch_dict['map_mask'].cuda().bool() # [b, map_element_num-Max, 20, 5]   [b, map_element_num-Max, 20]
        agent_feats = self.agent_net(agent_polylines, agent_polylines_mask) # [b,all_n-Max, d_agent]
        map_feats = self.map_net(map_polylines, map_polylines_mask) # [b, map_element_num-Max,d_map]
        
        rpe, rpe_mask = batch_dict['rpe'].cuda(), batch_dict['rpe_mask'].cuda().bool()
        batch_size, N, _, _= rpe.shape
        rpe_feats_valid = self.rpe_net(rpe[rpe_mask])  # (N, C)
        rpe_feats = rpe_feats_valid.new_zeros(batch_size, N, N, rpe_feats_valid.shape[-1])
        rpe_feats[rpe_mask] = rpe_feats_valid
        
        agent_mask = (agent_polylines_mask.sum(dim=-1) > 0)  
        map_mask = (map_polylines_mask.sum(dim=-1) > 0)  
        agent_feats, map_feat = self.fusion_net(agent_feats, agent_mask, map_feats, map_mask, rpe_feats, rpe_mask) # [b,all_n-Max, d_agent]
        
        plan_traj, plan_traj_mask = batch_dict['plan_feat'].cuda(), batch_dict['plan_mask'].cuda().bool()
        agent_feats, gate = self.plan_net(agent_feats, plan_traj, plan_traj_mask) # ?
        
        tar_candidate, candidate_mask = batch_dict['tar_candidate'].cuda(), batch_dict['candidate_mask'].cuda().bool() # [b,all_n-Max,Max-N-Max, 2]   [b,all_n-Max,Max-N-Max]
        target_gt = batch_dict['gt_preds'][:, :, -1, :2].cuda() # [b, all_n-Max, 2]
        target_gt = target_gt.view(target_gt.shape[0], target_gt.shape[1], 1, 2) # [b, all_n-Max, 1, 2]
        target_probs, pred_targets, pred_offsets, param, param_with_gt, traj_probs = self.traj_decoder(agent_feats, tar_candidate, target_gt, candidate_mask)
           
        # 由贝塞尔控制点反推轨迹
        # bezier_param = torch.cat([param, pred_targets], dim=-1) # B, N, m, (n_order+1)*2
        bezier_control_points = param.view(param.shape[0],
                                           param.shape[1],
                                           param.shape[2], -1, 2) # B, N, m, n_order+1, 2
        trajs = torch.matmul(self.mat_T, bezier_control_points) # B,N,m,future_steps,2
        
        # bezier_param_with_gt = torch.cat([param_with_gt, target_gt], dim=-1) # B, N, 1, (n_order+1)*2
        bezier_control_points_with_gt = param_with_gt.view(param_with_gt.shape[0],
                                                           param_with_gt.shape[1],
                                                           param_with_gt.shape[2], -1, 2) # B, N, 1, n_order+1, 2
        traj_with_gt = torch.matmul(self.mat_T, bezier_control_points_with_gt) # B,N,1,future_steps,2

        return {"target_probs": target_probs,
                "pred_offsets": pred_offsets,
                "traj_with_gt": traj_with_gt,
                "trajs": trajs,
                "traj_probs": traj_probs
               }  
        
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
        pred_mask = (input_dict['candidate_mask'].sum(dim=-1) > 0).cuda()
        # 计算目标点有关的指标
        target_top = 0
        gt_prob = input_dict['gt_candts'].cuda()[pred_mask]
        if gt_prob.dim() == 3:
            gt_prob = gt_prob.squeeze(dim=-1)
        batch_size, label_num = gt_prob.size()
        gt_label = torch.argmax(gt_prob, -1)
        target_prob = output_dict['target_probs'][pred_mask]
        
        max_probs, max_indices = torch.max(target_prob, dim=1)
        # 计算target的准确率
        for i in range(batch_size):
            if gt_label[i] ==max_indices[i]:
                target_top += 1
                
        target_gt = input_dict["gt_preds"].cuda()[pred_mask][:, -1, :2]
        target_candidate = input_dict["tar_candidate"].cuda()[pred_mask]
        offset = output_dict['pred_offsets'][pred_mask]
        target_pred = target_candidate[torch.arange(batch_size), max_indices] + offset[torch.arange(batch_size), max_indices]
        # 计算target的FDE
        rmse = torch.linalg.norm(target_pred - target_gt, dim=-1)
        target_fde = torch.sum(rmse)
     

        # 计算轨迹有关的指标
        score = output_dict['traj_probs'][pred_mask]
        traj_max_probs, traj_max_indices = torch.max(score, dim=1)
        trajs = output_dict['trajs'][pred_mask]
        traj_pred = trajs[torch.arange(batch_size), traj_max_indices].view(batch_size, 50, 2)
        traj_gt = input_dict["gt_preds"].cuda()[pred_mask]
        
        # 计算traj的ADE
        squared_distance = torch.sum((traj_pred - traj_gt) ** 2, dim=2)
        distance = torch.sqrt(squared_distance)
        traj_ade = torch.mean(distance, dim=1).sum()
       
        # 计算tarj的FDE
        fde =  torch.linalg.norm(traj_pred[:, -1, :] - traj_gt[:, -1, :], dim=-1)
        traj_fde = torch.sum(fde)
            
        return batch_size, target_top, target_fde, traj_ade, traj_fde