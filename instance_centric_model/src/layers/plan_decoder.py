import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_mlp import ResMLP



class PlanDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_order=7, m=50, refpath_dim = 64,embed_dim = 32):
        super().__init__()
        self.m = m
        self.refpath_dim = refpath_dim



        self.motion_estimator_layer = nn.Sequential(
            ResMLP(input_size + refpath_dim + embed_dim, hidden_size, hidden_size),
            nn.Linear(hidden_size, (n_order+1)*2) # n阶贝塞尔曲线，有n+1个控制点
        )
        
        self.traj_prob_layer = nn.Sequential(
            ResMLP(input_size + (n_order+1)*2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.vel_emb = nn.Parameter(torch.Tensor(1, 3, embed_dim))


    def forward(self,ego_feat, ego_refpath_feats, ego_vel_mode):
        '''
        input:
            - ego_feat: B,D
            - ego_refpath_feats: B, 64
            - ego_vel_mode: B
        output:
            - param B,3,(n_order+1)*2
            - traj_prob_tensor B,3
            - param_with_gt  B, (norder+1)*2
        '''
        B, _ = ego_refpath_feats.shape
        gt_idx= ego_vel_mode - 1
        all_gt_refpath = torch.zeros(B,3) #B,3
        batch_idx = torch.arange(B)
        all_gt_refpath[batch_idx, gt_idx] = 1 # B,3

        feats_cand = torch.cat([ego_feat, ego_refpath_feats],dim=-1) # B, D+64
        param_input = torch.cat([feats_cand.unsqueeze(1).repeat_interleave(repeats=3,dim=1), self.vel_emb.repeat(B,1,1)], dim=-1)# B,3,D+64     B,3,32 -> B,3,D+64+32
        # 1. 根据速度embed之后的refpath生成traj
        param = self.motion_estimator_layer(param_input) # B,3,D+64+32 -> B,3,(n_order+1)*2

  
        prob_input = torch.cat([ego_feat.unsqueeze(1).repeat(1,3,1), param],dim=-1) # B,3,D+(n_order+1)*2
        # 2.给traj打分
        traj_prob_tensor = self.traj_prob_layer(prob_input).squeeze(-1) # B,3


        param_with_gt = param[all_gt_refpath==1] # B, (norder+1)*2
        return param, traj_prob_tensor, param_with_gt