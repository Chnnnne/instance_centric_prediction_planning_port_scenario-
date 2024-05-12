import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,):
        """
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(Loss, self).__init__()
        self.lambda1 = 1
        self.lambda2 = 0.1
        self.lambda3 = 1
        self.lambda4 = 1

        self.temper = 0.01
        self.d_safe = 3.0
    
    def forward(self, input_dict, output_dict, epoch=1):
        loss = 0.0
        pred_mask = (input_dict['candidate_mask'].sum(dim=-1) > 0).cuda() # B,N,M->B, N  仅标志哪个agent是有效的，不标志refpath有效
        # 1、target_loss
        gt_probs = input_dict['gt_candts'].float().cuda() # B, N, M     

        gt_probs = gt_probs[pred_mask] # S, M   s个agent 每个agentM个refpath     
        pred_probs = output_dict['cand_refpath_probs'][pred_mask] # B, N, M -> S, M    s个agent，m个预测点
        pred_num = pred_probs.shape[0] # S
        cls_loss = F.binary_cross_entropy(pred_probs, gt_probs, reduction='sum')/pred_num 
        
        # gt_tar_offset = input_dict["gt_tar_offset"].cuda() # B, N ,2
        # try:
        #     gt_tar_offset = gt_tar_offset[pred_mask] # S, 2
        # except Exception as e:
        #     print('产生错误了:',e)
        #     print(f"pred_mask.shape: {pred_mask.shape}")
        #     print(f"gt_tar_offset: {gt_tar_offset.shape}")
        #     print("pred_mask:")
        #     print(pred_mask)
        #     print(torch.isfinite(gt_tar_offset))
        #     print(torch.isnan(gt_tar_offset))
        #     print("gt_tar_offset:")
        #     print(gt_tar_offset)

        # gt_idx = gt_probs.nonzero()[:pred_num] 
        # pred_offsets = output_dict['pred_offsets'][pred_mask] # S, M, 2
        # pred_offsets = pred_offsets[gt_idx[:, 0], gt_idx[:, 1]] # S, 2
        # offset_loss = F.smooth_l1_loss(pred_offsets, gt_tar_offset, reduction='sum')/pred_num # 只算所有候选中离gt最近的候选tar point的gt offset和预测出来的改哦point的offset算loss
        
        # 2、motion reg loss
        traj_with_gt = output_dict['traj_with_gt'].squeeze(2)[pred_mask] #B,N,1,50,2 -> B,N,50,2-> S, 50, 2
        gt_trajs = input_dict['gt_preds'].cuda()[pred_mask] #B,N,50,2 -> S, 50, 2
        reg_loss = F.smooth_l1_loss(traj_with_gt, gt_trajs, reduction="sum")/pred_num
        
        # 3、score_loss
        pred_trajs = output_dict['trajs'][pred_mask] # B,N,M,50,2 -> S, M, 50, 2
        S, m, horizon, dim = pred_trajs.shape
        pred_trajs = pred_trajs.view(S, m , horizon*dim) # S,M, 100
        gt_trajs = gt_trajs.view(S, horizon*dim)# S, 50, 2 -> S, 100
        score_gt = F.softmax(-self.distance_metric(pred_trajs,  gt_trajs)/self.temper, dim=-1).detach()# S,m
        score_loss = F.binary_cross_entropy(output_dict['traj_probs'][pred_mask], score_gt, reduction='sum')/pred_num  # SM 
        #由于要对 【模型对每条预测轨迹的打分器】 进行训练，因此我们就需要(打分器对每条轨迹的打分(其实是概率值))以及(真值)作为计算Loss的输入

        
        if epoch > 10:
            pred_trajs_t = pred_trajs.view(S, m, horizon, dim) #S,M,50,2
            plan_traj = input_dict["plan_feat"].cuda()[pred_mask][:, :, :2] # S, 50, 2
            plan_traj_mask = input_dict['plan_mask'].cuda()[pred_mask].bool() # S, 50
  
            distances = torch.sqrt(torch.sum((pred_trajs_t - plan_traj.unsqueeze(1))**2, dim=-1)) # S, m, 50
            masked_distances = distances.masked_fill(~plan_traj_mask.unsqueeze(1), 1000) # S, m, 50
            min_distances = torch.min(masked_distances, dim=2)[0] # S, m

            w_min_distances = output_dict['traj_probs'][pred_mask] * min_distances  # S, m
            min_distances_sum = torch.sum(w_min_distances, dim=-1)
            min_distances_sum = torch.clamp(min_distances_sum, max=self.d_safe)
            safety_loss = -torch.mean(min_distances_sum)

            loss = self.lambda1 * (cls_loss + 0) + self.lambda2 * reg_loss + self.lambda3 * score_loss + self.lambda4 * safety_loss
            loss_dict = {"ref_cls_loss": self.lambda1*cls_loss,
                        #  "tar_offset_loss": self.lambda1*offset_loss,
                         "traj_loss": self.lambda2*reg_loss,
                         "score_loss": self.lambda3*score_loss,
                         "safety_loss": self.lambda4 * safety_loss
                        }
        else:
            loss = self.lambda1 * (cls_loss + 0) + self.lambda2 * reg_loss + self.lambda3 * score_loss
            loss_dict = {"ref_cls_loss": self.lambda1*cls_loss,
                        #  "tar_offset_loss": self.lambda1*offset_loss,
                         "traj_loss": self.lambda2*reg_loss,
                         "score_loss": self.lambda3*score_loss}
        return loss, loss_dict
    
    # def forward(self, input_dict, output_dict, epoch=1):
    #     loss = 0.0
    #     pred_mask = (input_dict['candidate_mask'].sum(dim=-1) > 0).cuda() # B,N,M->B, N  仅标志哪个agent是有效的，不标志refpath有效
    #     # 1、target_loss
    #     gt_probs = input_dict['gt_candts'].float().cuda() # B, N, M     32 12 407

    #     gt_probs = gt_probs[pred_mask] # S, M   s个agent 每个agentM个refpath     
    #     pred_probs = output_dict['target_probs'][pred_mask] # B, N, M -> S, M    s个agent，m个预测点
    #     pred_num = pred_probs.shape[0] # S
    #     cls_loss = F.binary_cross_entropy(pred_probs, gt_probs, reduction='sum')/pred_num 
        
    #     gt_tar_offset = input_dict["gt_tar_offset"].cuda() # B, N ,2
    #     try:
    #         gt_tar_offset = gt_tar_offset[pred_mask] # S, 2
    #     except Exception as e:
    #         print('产生错误了:',e)
    #         print(f"pred_mask.shape: {pred_mask.shape}")
    #         print(f"gt_tar_offset: {gt_tar_offset.shape}")
    #         print("pred_mask:")
    #         print(pred_mask)
    #         print(torch.isfinite(gt_tar_offset))
    #         print(torch.isnan(gt_tar_offset))
    #         print("gt_tar_offset:")
    #         print(gt_tar_offset)

    #     gt_idx = gt_probs.nonzero()[:pred_num] 
    #     pred_offsets = output_dict['pred_offsets'][pred_mask] # S, M, 2
    #     pred_offsets = pred_offsets[gt_idx[:, 0], gt_idx[:, 1]] # S, 2
    #     offset_loss = F.smooth_l1_loss(pred_offsets, gt_tar_offset, reduction='sum')/pred_num # 只算所有候选中离gt最近的候选tar point的gt offset和预测出来的改哦point的offset算loss
        
    #     # 2、motion reg loss
    #     traj_with_gt = output_dict['traj_with_gt'].squeeze(2)[pred_mask] # S, 50, 2
    #     gt_trajs = input_dict['gt_preds'].cuda()[pred_mask] # S, 50, 2
    #     reg_loss = F.smooth_l1_loss(traj_with_gt, gt_trajs, reduction="sum")/pred_num
        
    #     # 3、score_loss
    #     pred_trajs = output_dict['trajs'][pred_mask] # S, m, 50, 2
    #     S, m, horizon, dim = pred_trajs.shape
    #     pred_trajs = pred_trajs.view(S, m , horizon*dim)
    #     gt_trajs = gt_trajs.view(S, horizon*dim)
    #     score_gt = F.softmax(-self.distance_metric(pred_trajs,  gt_trajs)/self.temper, dim=-1).detach()
    #     score_loss = F.binary_cross_entropy(output_dict['traj_probs'][pred_mask], score_gt, reduction='sum')/pred_num

        
    #     if epoch > 10:
    #         pred_trajs_t = pred_trajs.view(S, m, horizon, dim)
    #         plan_traj = input_dict["plan_feat"].cuda()[pred_mask][:, :, :2] # S, 50, 2
    #         plan_traj_mask = input_dict['plan_mask'].cuda()[pred_mask].bool() # S, 50
  
    #         distances = torch.sqrt(torch.sum((pred_trajs_t - plan_traj.unsqueeze(1))**2, dim=-1)) # S, m, 50
    #         masked_distances = distances.masked_fill(~plan_traj_mask.unsqueeze(1), 1000) # S, m, 50
    #         min_distances = torch.min(masked_distances, dim=2)[0] # S, m

    #         w_min_distances = output_dict['traj_probs'][pred_mask] * min_distances  # S, m
    #         min_distances_sum = torch.sum(w_min_distances, dim=-1)
    #         min_distances_sum = torch.clamp(min_distances_sum, max=self.d_safe)
    #         safety_loss = -torch.mean(min_distances_sum)

    #         loss = self.lambda1 * (cls_loss + offset_loss) + self.lambda2 * reg_loss + self.lambda3 * score_loss + self.lambda4 * safety_loss
    #         loss_dict = {"tar_cls_loss": self.lambda1*cls_loss,
    #                      "tar_offset_loss": self.lambda1*offset_loss,
    #                      "traj_loss": self.lambda2*reg_loss,
    #                      "score_loss": self.lambda3*score_loss,
    #                      "safety_loss": self.lambda4 * safety_loss
    #                     }
    #     else:
    #         loss = self.lambda1 * (cls_loss + offset_loss) + self.lambda2 * reg_loss + self.lambda3 * score_loss
    #         loss_dict = {"tar_cls_loss": self.lambda1*cls_loss,
    #                      "tar_offset_loss": self.lambda1*offset_loss,
    #                      "traj_loss": self.lambda2*reg_loss,
    #                      "score_loss": self.lambda3*score_loss}
    #     return loss, loss_dict
    
     
    def distance_metric(self, traj_candidate: torch.Tensor, traj_gt: torch.Tensor):
        """
        # S,m, 100
        # S,100
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
