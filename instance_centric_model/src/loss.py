import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,args):
        """
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(Loss, self).__init__()
        self.args = args
        self.lambda1 = 1 # cls
        self.lambda2 = 0.1 # traj_reg_loss
        self.lambda3 = 1 # traj_score_loss
        # self.lambda4 = 1
        self.lambda4 = 3 # irl_loss
        self.lambda5 = 2 # weight_reg

        self.temper = 0.01
        self.d_safe = 3.0
    
    def forward(self, input_dict, output_dict, epoch=1):
        loss = 0.0
        candidate_mask = input_dict['candidate_mask'].cuda() # B,N,M
        all_candidate_mask = output_dict['all_candidate_mask'].cuda() # B,N,3M
        pred_mask = (candidate_mask.sum(dim=-1) > 0).cuda() # B, N  

        s_candidate_mask = candidate_mask[pred_mask].cuda() # B,N,M+ B,N -> S, M
        s_all_candidate_mask = all_candidate_mask[pred_mask].cuda() # B,N,3M + B,N -> S,3M
        # 1、ref path cls loss
        gt_probs = input_dict['gt_candts'].float().cuda() # B, N, M     

        gt_probs = gt_probs[pred_mask] # S, M   s个agent, 每个agent M个refpath     
        pred_probs = output_dict['cand_refpath_probs'][pred_mask] # B, N, M -> S, M （s个agent，m个预测点） 被mask住的数据，model预测已调整为0，
        pred_num = pred_probs.shape[0] # S
        cls_loss = F.binary_cross_entropy(pred_probs, gt_probs, reduction='sum')/pred_num  # 对于被mask的数据项，根据BSE定义，得单项loss为0，因此此处的loss无需再做mask
        

        # 2、motion reg loss
        traj_with_gt = output_dict['traj_with_gt'].squeeze(2)[pred_mask] #B,N,1,50,2 -> B,N,50,2-> S, 50, 2
        gt_trajs = input_dict['gt_preds'].cuda()[pred_mask] #B,N,50,2 -> S, 50, 2
        reg_loss = F.smooth_l1_loss(traj_with_gt, gt_trajs, reduction="sum")/pred_num
        
        # 3、score_loss
        #由于要对 【模型对每条预测轨迹的打分器】 进行训练，因此我们就需要(打分器对每条轨迹的打分(其实是概率值))以及(真值)作为计算Loss的输入
        pred_trajs = output_dict['trajs'][pred_mask] # B,N,3M,50,2 -> S, 3M, 50, 2
        S, m, horizon, dim = pred_trajs.shape
        pred_trajs = pred_trajs.view(S, m, horizon*dim) # S,3M, 100
        gt_trajs = gt_trajs.view(S, horizon*dim)# S, 50, 2 -> S, 100
        score_gt = self.masked_softmax(vector=-self.distance_metric(pred_trajs,  gt_trajs)/self.temper, mask=s_all_candidate_mask).detach() # S,3m  + S,3m
        score_loss = F.binary_cross_entropy(output_dict['traj_probs'][pred_mask], score_gt, reduction='sum')/pred_num  # S,3M + S,3M   model输出的traj_probs对于mask数据已调整为0，score_gt对mask的数据也也做处理，因此此处可直接算BCE

        # 4. planning_loss
        B, _, _ = input_dict['ego_gt_traj'].shape
        # 4.1 plannig ref path cls loss
        ego_gt_probs = input_dict['ego_gt_cand'].float().cuda() # B, M     
        plan_pred_probs = output_dict['plan_cand_refpath_probs'] # B, M
        plan_cls_loss = F.binary_cross_entropy(plan_pred_probs, ego_gt_probs, reduction='sum')/B

        # 4.2 planning traj reg loss
        plan_traj_with_gt = output_dict['plan_traj_with_gt']# B, 50, 2
        ego_gt_traj = input_dict['ego_gt_traj'].cuda() # B, 50, 2
        plan_reg_loss = F.smooth_l1_loss(plan_traj_with_gt, ego_gt_traj, reduction="sum")/B

        # 4.3 planning traj prob loss
        plan_M = output_dict['plan_trajs'].shape[1]
        plan_pred_trajs = output_dict['plan_trajs'].view(B,plan_M,horizon*dim) # B,3M,50,2 -> B,3M,100
        plan_all_candidate_mask = output_dict['plan_all_candidate_mask'] # B,3M
        ego_gt_traj = ego_gt_traj.view(B, horizon*dim)# B,100
        # plan_score_gt = F.softmax(-self.distance_metric(plan_pred_trajs, ego_gt_traj)/self.temper, dim=-1).detach() # B,3,100 + B,100 -> B,3 -> B,3
        plan_score_gt = self.masked_softmax(vector=-self.distance_metric(plan_pred_trajs, ego_gt_traj)/self.temper,mask=plan_all_candidate_mask).detach() # B,3M,100 + B,100 -> B,3M   + mask=B,3M -> B,3M
        plan_score_loss = F.binary_cross_entropy(output_dict['plan_traj_probs'], plan_score_gt, reduction='sum')/B # B,3M + B,3M

        # irl loss
        scores, weights = output_dict['scores'], output_dict['weights']# scores B,3M    weight B8
        min_idx = self.get_closest_traj_idx(output_dict['plan_trajs'], input_dict['ego_gt_traj'].to(output_dict['plan_trajs'].device)) # B,3M,50,2  + B,50,2->B
        if self.args.train_part == "back" or self.args.train_part == "joint":
            irl_loss = F.cross_entropy(scores, min_idx)
            weights_regularization = torch.square(weights).mean()
        else:
            irl_loss = torch.tensor(0).cuda()
            weights_regularization = torch.tensor(0).cuda()

        if self.args.train_part == "back":
            loss = self.lambda4 * irl_loss + self.lambda5 * weights_regularization
        elif self.args.train_part == "front":
            loss = self.lambda1 * cls_loss + self.lambda2 * reg_loss + self.lambda3 * score_loss  + self.lambda1*plan_cls_loss + self.lambda2 * plan_reg_loss + self.lambda3 * plan_score_loss 
        else:# joint
            loss = self.lambda1 * cls_loss + self.lambda2 * reg_loss + self.lambda3 * score_loss  + self.lambda2 * plan_reg_loss + self.lambda4 * irl_loss + self.lambda5 * weights_regularization

        loss_dict = {"ref_cls_loss": self.lambda1*cls_loss,
                        "traj_loss": self.lambda2*reg_loss,
                        "score_loss": self.lambda3*score_loss,

                        "plan_cls_loss":self.lambda1*plan_cls_loss,
                        "plan_reg_loss": self.lambda2*plan_reg_loss,
                        "plan_score_loss": self.lambda3*plan_score_loss,

                        "irl_loss": self.lambda4 * irl_loss,
                        "weights_regularization": self.lambda5 * weights_regularization
                        }
        return loss, loss_dict
    
    def get_closest_traj_idx(self, plan_trajs, gt_trajs):
        '''
        - plan_trajs: B,3M,50,2   
        - gt_trajs: B,50,2
        '''
        B,_,T,_ = plan_trajs.shape
        dists = torch.norm(plan_trajs - gt_trajs[:,None,:,:], dim=-1) # B,3M,50,2 -》 B,3M,50 
        dists = torch.linspace(0.7,1.3, T,device=dists.device) * dists
        min_idx = torch.argmin(dists.sum(-1),dim=-1) # B
        return min_idx



     
    def distance_metric(self, traj_candidate: torch.Tensor, traj_gt: torch.Tensor):
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
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value) # 将BNM中没有被mask住的位置置为负无穷
                result = F.softmax(masked_vector, dim=dim) # 做softmax得到概率
                result = result.masked_fill((1 - mask).bool(), 0.0) # BNM没被mask住的位置概率置为0
        return result
