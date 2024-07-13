import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_mlp import ResMLP

class TrajDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_order=7, m=50, refpath_dim = 64,embed_dim = 32):
        super().__init__()
        self.m = m
        # monomial basis polynomial
        # a3, a2, a1, a0.   x=a3t^3 + a2t^2 + a1t^1 + a0
        # b3, b2, b1, b0.   y=b3t^3 + b2t^2 + b1t^1 + b0
        mbp_coff_num = 8

        self.cand_refpath_prob_layer = nn.Sequential(
            ResMLP(input_size + refpath_dim, hidden_size, hidden_size),# 128+64
            nn.Linear(hidden_size, 1)
        )


        self.motion_estimator_layer = nn.Sequential(
            ResMLP(input_size + refpath_dim + embed_dim, hidden_size, hidden_size), # 128+64+32
            nn.Linear(hidden_size, mbp_coff_num) # n阶贝塞尔曲线，有n+1个控制点
        )
        
        self.traj_prob_layer = nn.Sequential(
            ResMLP(input_size + mbp_coff_num, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=2)
        )

        # 可学习的速度嵌入向量
        self.vel_emb = nn.Parameter(torch.Tensor(1, 1, 3, embed_dim))


    def forward(self, feats, refpath_feats, gt_refpath, gt_vel_mode, candidate_mask=None):
        """
        input:
            - feats: B,N,D                                                                         
            - candidate_refpaths_cords:   B,N,M,20,2                                                          
            - candidate_refpaths_vecs: B,N,M,20,2  N个agent ， 每个agent共M的采样点（M是最大的）             
            - gt_refpath:  B, N, M                真值refpath one hot                                 
            - gt_vel_mode: B, N
            - candidate_mask: B, N, M             因为每个agent的候选path数量不同，因此 M标记哪些是和不是refpath   
        
        output:
            - all_gt_refpath B,N,3M,     标志候选refpath(速度扩充后)的真值onehot
            - all_candidate_mask: B, N, 3M  标志候选refpath（速度扩充后）的有效性
            - cand_refpath_probs:   B,N,M 
            - param:    B,N,3M,100
            - traj_probs:       B, N, 3M
            - param_with_gt:     B,N,1,100
        """
        #可视化验证
        # self.vis_debug(candidate_refpaths_cords=candidate_refpaths_cords, candidate_refpaths_vecs=candidate_refpaths_vecs,gt_refpath=gt_refpath,candidate_mask=candidate_mask,batch_dict=batch_dict)
        B, N, M = gt_refpath.shape 
        gt_idx = (torch.argmax(gt_refpath, dim=-1)*3+gt_vel_mode - 1) # BNM->BN-> *3+- ->BN (gt_refpath全零则gt_idx对应为-1，后续算loss会无视)
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)  # 生成 [B, N] 形状的批次索引
        sequence_indices = torch.arange(N).unsqueeze(0).expand(B, N)  # 生成 [B, N] 形状的序列索引
        all_gt_refpath = torch.zeros(B,N,3*M) # B,N,3M
        all_gt_refpath[batch_indices, sequence_indices, gt_idx] = 1 # 标志速度embedding扩充之后的真值
        all_candidate_mask = candidate_mask.repeat_interleave(repeats=3, dim= -1) # B,N,M -> B,N,3M   标志速度embedding扩充之后的cand mask

        # refpath_feats = self.refpath_encoder(torch.cat([candidate_refpaths_cords.reshape(B,N,M,-1), candidate_refpaths_vecs.reshape(B,N,M,-1)],dim=-1))# B,N,M,40 +40  ->B,N,M,64
        agent_feats_repeat = feats.unsqueeze(2).repeat(1, 1, M, 1) # B, N, M, D
        feats_cand = torch.cat([agent_feats_repeat, refpath_feats], dim=-1) # B,N,M, D+64
        # 1. refpath打分
        prob_tensor = self.cand_refpath_prob_layer(feats_cand).squeeze(-1) # B,N,M, D+64 -> B,N,M 
        cand_refpath_probs = self.masked_softmax(prob_tensor, candidate_mask, dim=-1) # B,N,M + B,N,M -> B,N,M 空数据被mask为概率0

        # [B,N,M, D + 64+ embed]   *  3

        # 2. 根据refpath生成traj
        param_input = torch.cat([feats_cand.repeat_interleave(repeats=3,dim=2),self.vel_emb.repeat(B,N,M,1)], dim=-1)# B,N,3m,D+64+emd_dim

        param = self.motion_estimator_layer(param_input) # B,N,3M,8 空的数据也会预测轨迹，可由下面的prob做mask，因为prob为0的轨迹忽略,输出轨迹也没事


        # 3. 给traj打分
        prob_input = torch.cat([agent_feats_repeat.repeat(1,1,3,1), param], dim=-1) # B, N, 3M, D + 8
        traj_prob_tensor = self.traj_prob_layer(prob_input).squeeze(-1) # B, N, 3M   打分 空的parm和特征数据也会打分，做了softmax但没做mask，因此没mask的位置也会有概率评分
        traj_probs = self.masked_softmax(traj_prob_tensor, all_candidate_mask, dim = -1) # B,N,3M + B,N,3M
 
        
        # 预测轨迹(teacher_force)
        param_with_gt = param[all_gt_refpath==1].unsqueeze(1).reshape(B,N,1,-1) #  B,N,3M,8 -> [B*N, 8] -> [B,N,1,8]
  
        return cand_refpath_probs, param, traj_probs, param_with_gt,all_candidate_mask
    
    def forward_origin(self, feats, tar_candidate, target_gt, candidate_mask=None):
        """
        feats: B,N,D                                                                           [b,all_n-Max, d_agent]
        tar_candidate: B, N, M, 2           N个agent ， 每个agent共M的采样点（M是最大的）           [b,all_n-Max,Max-N-Max, 2]  
        target_gt:  B, N, 1, 2              每个agent一个真值点                                   [b, all_n-Max, 1, 2]
        candidate_mask: B, N, M             因为每个agent的采样点数量不同，因此 M标记哪些是和不是      [b,all_n-Max,Max-N-Max]
        """
        
        B, N, M, _ = tar_candidate.shape
        feats_repeat = feats.unsqueeze(2).repeat(1, 1, M, 1) # B, N, M, D

        # stack the target candidates to the end of input feature
        feats_tar = torch.cat([feats_repeat, tar_candidate], dim=-1) # B, N, M, D+2
        # compute probability for each candidate
        prob_tensor = self.taregt_prob_layer(feats_tar).squeeze(-1) # B,N,M




        # 1 feat和候选参考点送入点预测期得到概率、offset
        target_probs = self.masked_softmax(prob_tensor, candidate_mask, dim=-1) # B, N, M
        
        tar_offsets = self.target_offset_layer(feats_tar) # B, N, M, 2
        




        # 获取topk
        m = min(target_probs.shape[2], self.m)
        _, topk_indices = target_probs.topk(m, dim=2) # B, N, m50
        tar_indices = topk_indices.unsqueeze(-1).expand(topk_indices.shape[0], 
                                                        topk_indices.shape[1], 
                                                        topk_indices.shape[2], 
                                                        tar_candidate.shape[-1]) # B,N,m,1->  B,N,m,2
        target_pred_se = torch.gather(tar_candidate, dim=2, index=tar_indices) # B, N, m, 2    inputB, N, M, 2    index B,N,m,2
        offset_pred_se = torch.gather(tar_offsets, dim=2, index=tar_indices) # B, N, m, 2
        # 组合feat
        target_pred = target_pred_se + offset_pred_se # B, N, m, 2








        feat_indices = topk_indices.unsqueeze(-1).expand(topk_indices.shape[0], 
                                                        topk_indices.shape[1], 
                                                        topk_indices.shape[2], 
                                                        feats_repeat.shape[-1])
        feats_traj = torch.gather(feats_repeat, dim=2, index=feat_indices) # B, N, m, D
        # 组合feat
        param_input = torch.cat([feats_traj, target_pred], dim=-1) # B, N, m, D+2
        # 2 topk的target和feat送入轨迹预测器得到贝塞尔控制点
        param = self.motion_estimator_layer(param_input) # B,N,m,n_order*2




        # 3 贝塞尔控制点和feat送入,得到轨迹打分
        prob_input = torch.cat([feats_traj, param], dim=-1) # B, N, m， D + n_order*2
        traj_probs = self.traj_prob_layer(prob_input).squeeze(-1) # B, N, m 打分
        
        # 预测轨迹(teacher_force)
        # 2.1 真值点送入轨迹预测器得到真值点预测得到的贝塞尔控制点
        feat_traj_with_gt = torch.cat([feats.unsqueeze(2), target_gt], dim=-1) # B, N, 1, D+2
        param_with_gt = self.motion_estimator_layer(feat_traj_with_gt) # B,N,1,n_order*2
        
        return target_probs, target_pred, tar_offsets, param, param_with_gt, traj_probs
    
    
    def masked_softmax(self, vector, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32):
        # vector BN3m = mask  BNM
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