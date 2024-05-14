import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_mlp import ResMLP

class TrajDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_order=7, m=50):
        super().__init__()
        self.m = m
        # self.taregt_prob_layer = nn.Sequential(
        #     ResMLP(input_size + 2, hidden_size, hidden_size),
        #     nn.Linear(hidden_size, 1)
        # )
        self.cand_refpath_prob_layer = nn.Sequential(
            ResMLP(input_size + 20*2 + 20*2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1)
        )
        # self.target_offset_layer = nn.Sequential(
        #     ResMLP(input_size + 2, hidden_size, hidden_size),
        #     nn.Linear(hidden_size, 2)
        # )
        # self.motion_estimator_layer = nn.Sequential(
        #     ResMLP(input_size + 2, hidden_size, hidden_size),
        #     nn.Linear(hidden_size, (n_order+1)*2) # n阶贝塞尔曲线，有n+1个控制点
        # )
        self.motion_estimator_layer = nn.Sequential(
            ResMLP(input_size + 20*2 + 20*2, hidden_size, hidden_size),
            nn.Linear(hidden_size, (n_order+1)*2) # n阶贝塞尔曲线，有n+1个控制点
        )
        
        self.traj_prob_layer = nn.Sequential(
            ResMLP(input_size + (n_order+1)*2, hidden_size, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=2)
        )

    # def vis_debug(self, candidate_refpaths_cords,candidate_refpaths_vecs,gt_refpath, candidate_mask, batch_dict):
    #     pred_mask = (candidate_mask.sum(dim=-1) > 0).cuda() # B,N,M->B, N  仅标志哪个agent是有效的，不标志refpath有效
    #     idx_mask = candidate_mask.sum(dim=-1).cuda() # B,N,M->B, N  
    #     idx_mask = idx_mask[pred_mask] # S

    #     agent_hiss = batch_dict['agent_feats'][pred_mask][:,:,:2] # B,N,20,13, -> S, 20, 2
    #     oris = batch_dict['agent_ctrs'][pred_mask] # B,N,2 -> S,2
    #     refpath_cords = candidate_refpaths_cords[pred_mask] #B,N,M,20,2 -> S, M, 20, 2
    #     gt_preds = batch_dict['gt_preds'][pred_mask]# B,N,50,2 -> S, 50, 2


    #     gt_idx = gt_refpath[pred_mask] # S,M
    #     gt_idx = torch.argmax(gt_idx,axis = 1)#S
    #     S = refpath_cords.shape[0]
    #     import sys
    #     sys.path.append("/data/wangchen/instance_centric")
    #     print(sys.path)
    #     import common.plot_utils as plot_utils
    #     import pickle
    #     for i in range(S):
    #         tmp_dict = {"ori":oris[i], "candidate_refpaths":refpath_cords[i,:idx_mask[i]],"cand_gt_idx":gt_idx[i],"his_traj": agent_hiss[i],"fut_traj":gt_preds[i] }
    #         save_path = "/data/wangchen/instance_centric/tmp" + f'/{i}_{1}.pkl'
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(tmp_dict, f)
    #             print("$"*80)
    #         # plot_utils.draw_candidate_refpaths_with_his_fut(oris[i],candidate_refpaths=refpath_cords[i,:idx_mask[i]],
    #         #                                                 cand_gt_idx=gt_idx[i],his_traj=agent_hiss[i],fut_traj=gt_preds[i])
    #         print("draw one")

    #     print("finish vis debug")
    #     exit()
    #     # B = oris.shape[0]
    #     # for idx in range(B):
    #     #     his = agent_hiss[idx] # N, 20, 2
    #     #     ori = oris[idx] # N, 2
    #     #     refpath_cords = candidate_refpaths_cords[idx] # 
    #     #     refpath_vecs = candidate_refpaths_vecs[idx]
    #     #     gt_ref = gt_refpath[idx]
    #     #     mask = candidate_mask[idx]


    def forward(self, feats, candidate_refpaths_cords, candidate_refpaths_vecs, gt_refpath, candidate_mask=None,batch_dict = None):
        """
        input:
            - feats: B,N,D                                                                           [b,all_n-Max, d_agent]
            - candidate_refpaths_cords:   B,N,M,20,2                                                           [b,all_n-Max,Max-N-Max, 20, 2] 
            - candidate_refpaths_vecs: B,N,M,20,2  N个agent ， 每个agent共M的采样点（M是最大的）            [b,all_n-Max,Max-N-Max, 20, 2]  
            - gt_refpath:  B, N, M                真值refpath one hot                                 [b, all_n-Max, Max-N-Max]
            - candidate_mask: B, N, M             因为每个agent的采样点数量不同，因此 M标记哪些是和不是      [b,all_n-Max,Max-N-Max]
        
        output:
            - cand_refpath_probs:   B,N,M 
            - param:    B,N,M,(n_order+1)*2
            - traj_probs:       B, N, M
            - param_with_gt:     B,N,1,(n_order+1)*2
        """
        #可视化验证
        # self.vis_debug(candidate_refpaths_cords=candidate_refpaths_cords, candidate_refpaths_vecs=candidate_refpaths_vecs,gt_refpath=gt_refpath,candidate_mask=candidate_mask,batch_dict=batch_dict)
        B, N, M, _, _ = candidate_refpaths_vecs.shape # N为agent数量pad后值 M为refpath数量pad后值
        feats_repeat = feats.unsqueeze(2).repeat(1, 1, M, 1) # B, N, M, D
        feats_cand = torch.cat([feats_repeat, candidate_refpaths_cords.reshape(B,N,M,-1), candidate_refpaths_vecs.reshape(B,N,M,-1)], dim=-1) # B,N,M, D+40+40
        prob_tensor = self.cand_refpath_prob_layer(feats_cand).squeeze(-1) # B,N,M, D+40+40 -> B,N,M 很多空的D 4040送入预测器
        cand_refpath_probs = self.masked_softmax(prob_tensor, candidate_mask, dim=-1) # B,N,M + B,N,M -> B,N,M 空数据被mask为概率0

        param_input = feats_cand  # B,N,M, D+40+40
        param = self.motion_estimator_layer(param_input) # B,N,M,(n_order+1)*2 空的数据也会预测轨迹，可由下面的prob做mask，因为prob为0的轨迹忽略

        prob_input = torch.cat([feats_repeat, param], dim=-1) # B, N, M, D + (n_order+1)*2
        traj_prob_tensor = self.traj_prob_layer(prob_input).squeeze(-1) # B, N, M 打分 空的parm和特征数据也会打分，做了softmax但没做mask，因此没mask的位置也会有概率评分
        traj_probs = self.masked_softmax(traj_prob_tensor, candidate_mask, dim = -1) # B,N,M + B,N,M
        # 预测轨迹(teacher_force)
        # feat_traj_with_gt = torch.cat([feats.unsqueeze(2), target_gt], dim=-1) # B, N, 1, D+2
        # param_with_gt = self.motion_estimator_layer(feat_traj_with_gt) # B,N,1,(n_order+1)*2
        gt_idx = torch.argmax(gt_refpath, dim=-1).unsqueeze(-1)  # B, N, M  -> B, N, 1
        gt_idx = gt_idx.unsqueeze(-1).expand(-1,-1,-1, param.size(3)) # B, N, 1, (n_order+1)*2

        param_with_gt = torch.gather(param, dim=2, index=gt_idx)# B, N, 1, (n_order+1)*2 没做mask
        
        return cand_refpath_probs, param, traj_probs, param_with_gt
    
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
        # vector = mask = BNM
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