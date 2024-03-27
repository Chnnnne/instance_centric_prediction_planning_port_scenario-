import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_mlp import ResMLP

class TrajDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_order=7, m=50):
        super().__init__()
        self.m = m
        self.taregt_prob_layer = nn.Sequential(
            ResMLP(input_size + 2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.target_offset_layer = nn.Sequential(
            ResMLP(input_size + 2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 2)
        )
        self.motion_estimator_layer = nn.Sequential(
            ResMLP(input_size + 2, hidden_size, hidden_size),
            nn.Linear(hidden_size, (n_order+1)*2)
        )
        
        self.traj_prob_layer = nn.Sequential(
            ResMLP(input_size + (n_order+1)*2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=2)
        )
    def forward(self, feats, tar_candidate, target_gt, candidate_mask=None):
        """
        feats: B,N,D
        tar_candidate: B, N, M, 2
        target_gt:  B, N, 1, 2
        candidate_mask: B, N, M
        """
        
        B, N, M, _ = tar_candidate.shape
        feats_repeat = feats.unsqueeze(2).repeat(1, 1, M, 1)

        # stack the target candidates to the end of input feature
        feats_tar = torch.cat([feats_repeat, tar_candidate], dim=-1) # B, N, M, D+2
        # compute probability for each candidate
        prob_tensor = self.taregt_prob_layer(feats_tar).squeeze(-1) # B,N,M
        target_probs = self.masked_softmax(prob_tensor, candidate_mask, dim=-1) # B, N, M
        
        tar_offsets = self.target_offset_layer(feats_tar) # B, N, M, 2
        
        m = min(target_probs.shape[2], self.m)
        _, topk_indices = target_probs.topk(m, dim=2)
        tar_indices = topk_indices.unsqueeze(-1).expand(topk_indices.shape[0], 
                                                        topk_indices.shape[1], 
                                                        topk_indices.shape[2], 
                                                        tar_candidate.shape[-1])
        target_pred_se = torch.gather(tar_candidate, dim=2, index=tar_indices) # B, N, m, 2
        offset_pred_se = torch.gather(tar_offsets, dim=2, index=tar_indices) # B, N, m, 2
        
        target_pred = target_pred_se + offset_pred_se
        feat_indices = topk_indices.unsqueeze(-1).expand(topk_indices.shape[0], 
                                                        topk_indices.shape[1], 
                                                        topk_indices.shape[2], 
                                                        feats_repeat.shape[-1])
        feats_traj = torch.gather(feats_repeat, dim=2, index=feat_indices) # B, N, m, D
        param_input = torch.cat([feats_traj, target_pred], dim=-1) # B, N, m, D+2

        param = self.motion_estimator_layer(param_input) # B,N,m,n_order*2
        prob_input = torch.cat([feats_traj, param], dim=-1)
        traj_probs = self.traj_prob_layer(prob_input).squeeze(-1) # B, N, m
        
        # 预测轨迹(teacher_force)
        feat_traj_with_gt = torch.cat([feats.unsqueeze(2), target_gt], dim=-1) # B, N, 1, D+2
        param_with_gt = self.motion_estimator_layer(feat_traj_with_gt) # B,N,1,n_order*2
        
        return target_probs, target_pred, tar_offsets, param, param_with_gt, traj_probs
    
    
    def masked_softmax(self, vector, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32):
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