import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_mlp import ResMLP



class PlanDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_order=7, m=50, refpath_dim = 64,embed_dim = 32):
        super().__init__()
        self.m = m
        self.refpath_dim = refpath_dim

        self.cand_refpath_prob_layer = nn.Sequential(
            ResMLP(input_size + refpath_dim, hidden_size, hidden_size),# 128+64
            nn.Linear(hidden_size, 1)
        )

        self.motion_estimator_layer = nn.Sequential(
            ResMLP(input_size + refpath_dim + embed_dim, hidden_size, hidden_size),
            nn.Linear(hidden_size, (n_order+1)*2) # n阶贝塞尔曲线，有n+1个控制点
        )
        
        self.traj_prob_layer = nn.Sequential(
            nn.Linear(input_size + (n_order+1)*2, 1),
            nn.Softmax(dim=1)
        )
        # self.traj_prob_layer = nn.Sequential(
        #     ResMLP(input_size + (n_order+1)*2, hidden_size, hidden_size),
        #     nn.Linear(hidden_size, 1),
        #     nn.Softmax(dim=1)
        # )
        self.vel_emb = nn.Parameter(torch.Tensor(1, 3, embed_dim))


    def forward(self,ego_feat, ego_refpath_feats, ego_vel_mode, ego_cand_mask, ego_gt_cand):
        '''
        input:
            - ego_feat: B,D
            - ego_refpath_feats: B, M, 64
            - ego_vel_mode: B
            - ego_cand_mask: B,M
            - ego_gt_cand: B,M
        output:
            - cand_refpath_probs  B,M
            - param B,3,(n_order+1)*2
            - traj_prob_tensor B,3
            - param_with_gt  B, (norder+1)*2
        '''
        B, M,_ = ego_refpath_feats.shape
        gt_idx = ego_gt_cand.argmax(dim=-1) * 3 + ego_vel_mode - 1 # B

        all_gt_refpath = torch.zeros(B, 3*M) # B,3M
        all_gt_refpath[torch.arange(B), gt_idx] = 1


        all_candidate_mask = ego_cand_mask.repeat_interleave(repeats=3, dim=-1) # B,M -> B,3M




        feats_cand = torch.cat([ego_feat.unsqueeze(1).repeat(1, M, 1), ego_refpath_feats],dim=-1) # B,M,D + 64  

        # 1
        prob_tensor = self.cand_refpath_prob_layer(feats_cand).squeeze(-1) # # B,M,D + 64 ->  B,M
        cand_refpath_probs = self.masked_softmax(prob_tensor, ego_cand_mask, dim = -1) # B,M + B,M ->B,M
        
        # B,M,D+64 -> B,3M,D+64   cat  -> B,3M,D+64+emd_d
        param_input = torch.cat([feats_cand.repeat_interleave(repeats=3,dim=1), self.vel_emb.repeat(B,M,1)], dim=-1)

        # 2. 根据速度embed之后的refpath生成traj
        param = self.motion_estimator_layer(param_input) # B,3M,D+64+emd_d -> B,3M,(n_order+1)*2

  
        prob_input = torch.cat([ego_feat.unsqueeze(1).repeat(1, 3*M, 1), param],dim=-1) # B,3M,D+(n_order+1)*2
        # 3.给traj打分
        traj_prob_tensor = self.traj_prob_layer(prob_input).squeeze(-1) # B,3M
        traj_probs = self.masked_softmax(traj_prob_tensor, all_candidate_mask, dim=-1) # B,3M

        param_with_gt = param[all_gt_refpath==1] # B,3M,(n_order+1)*2       B, (norder+1)*2
        return cand_refpath_probs, param, traj_probs, param_with_gt, all_candidate_mask




    def masked_softmax(self, vector, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32):
        # vector B, 3m = mask  
        if mask is None:
            result = F.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(-1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = F.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
                result = result.masked_fill((1 - mask).bool(), 0.0)
            else:
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
                result = F.softmax(masked_vector, dim=dim)
                result = result.masked_fill((1 - mask).bool(), 0.0)
        return result