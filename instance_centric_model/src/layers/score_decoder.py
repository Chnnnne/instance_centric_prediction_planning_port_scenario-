import torch
import torch.nn as nn
from scipy.special import comb
from common.math_utils import bernstein_poly, bezier_derivative, bezier_curve



class ScoreDecoder(nn.Module):
    def __init__(self, n_order = 7, variable_cost=False):
        super(ScoreDecoder, self).__init__()
        self.n_order = n_order
        self._n_latent_features = 4
        self._variable_cost = variable_cost

        self.interaction_feature_encoder = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 256))
        self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features), nn.Sigmoid())
        
        # self.decoder = nn.Sequential(nn.Linear)
        # self.interaction_feature_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        # self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features), nn.Sigmoid())

        self.weights_decoder = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, self._n_latent_features+4), nn.Softplus())
        # self.weights_param = nn.Parameter(torch.Tensor(1, 1, 4))

    

    def get_ego_yaw_vel(self, ego_traj, plan_param, plan_all_candidate_mask):
        # 绝对坐标系下的yaw vel属性
        B,_,T,_ = ego_traj.shape
        t_values = torch.linspace(0, 1, T).cuda()
        vec_param = bezier_derivative(plan_param) # B,3M,(n_order+1), 2 -> B,3M,(n_order),2
        vec_vectors = bezier_curve(vec_param, t_values)/5 # B,3M,50,2
        vec_vectors = vec_vectors.clamp(-15,15) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1)
        vec_scaler = torch.norm(vec_vectors,dim=-1) # B,3M,50
        vx,vy = vec_vectors[...,0], vec_vectors[...,1]
        yaw_angle = torch.atan2(vy, vx)# B,3M,50
        ego_traj = torch.cat([ego_traj, yaw_angle.unsqueeze(-1), vec_scaler.unsqueeze(-1)],dim=-1) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1) # B, 3M, 50, 4
        return ego_traj



    def get_hardcoded_features(self, ego_traj, plan_param,plan_all_candidate_mask):
        '''
        相对坐标系下的编码特征
        input:
            - ego_traj: B, 3M, 50, 2
            - plan_param: B,3M,(n_order+1),2
            - plan_all_candidate_mask: B,3M
        output: 
            - features: B,3M,4
            - ego_traj: B,3M,50,4
        除以5的原因：是按照1s走的50个点而算出来的速度， 而实际情况是5s走了50个点，所以速度/5
        ''' 

        B,_,T,_ = ego_traj.shape

        t_values = torch.linspace(0, 1, T).cuda()
        #1 速度
        vec_param = bezier_derivative(plan_param) # B,3M,(n_order+1), 2 -> B,3M,(n_order),2
        vec_vectors = bezier_curve(vec_param, t_values)/5 # B,3M,50,2
        vec_vectors = vec_vectors.clamp(-15,15) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1)
        vec_scaler = torch.norm(vec_vectors,dim=-1) # B,3M,50

        #2 加速度
        acc_param = bezier_derivative(vec_param) # B,3M,(n_order),2 ->B,3M,(n_order - 1),2
        acc_vectors = bezier_curve(acc_param, t_values)/25 # B,3M,50,2
        acc_vectors = acc_vectors.clamp(-10,10) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1)
        acc_scaler = torch.norm(acc_vectors,dim=-1)# B,3M,50

        #3 加加速度
        jerk_param = bezier_derivative(acc_param) # B,3M,(n_order-1),2 -> B,3M,(n_order - 2),2
        jerk_vectors = bezier_curve(jerk_param, t_values)/125 # B,3M,50,2
        jerk_vectors = jerk_vectors.clamp(-10,10) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1)
        jerk_scaler = torch.norm(jerk_vectors,dim=-1)# B,3M,50

        #4 曲率
        # 曲率的公式：κ = |v_x * a_y - v_y * a_x| / (v_x^2 + v_y^2)^(3/2) B,3,50,2
        epsilon = 1e-4
        vx,vy = vec_vectors[...,0], vec_vectors[...,1]
        ax,ay = acc_vectors[...,0], acc_vectors[...,1]
        curvature = torch.abs(vx * ay - vy * ax)/((vx**2 + vy**2+epsilon)**1.5) # B,3M,50
        # 检查是否有nan值
        if torch.isnan(curvature).any():
            print("Curvature has NaN values:\n", curvature)
            print("vx:",vx)
            print("vy:",vy)
            print("ax:",ax)
            print("ay:",ay)

        curvature = torch.clamp(curvature, min=0, max=3)
        curvature = curvature * plan_all_candidate_mask.unsqueeze(-1)

        #5 横向加速度
        # 横向加速度的公式：a_lateral = v^2 * κ
        lateral_acceleration = vec_scaler**2 * curvature # B,3M,50
        lateral_acceleration = lateral_acceleration * plan_all_candidate_mask.unsqueeze(-1)
        #6 航向角
        # yaw_angle = torch.atan2(vy, vx)# B,3M,50

        v = -vec_scaler.mean(-1).clip(0, 15) / 15 # B,3M,50 -> B,3M
        a = acc_scaler.abs().mean(-1).clip(0, 4) / 4 # B,3M,50 -> B,3M
        j = jerk_scaler.abs().mean(-1).clip(0, 6) / 6 # B,3M,50 -> B,3M
        la = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5 # B,3M,50 -> # B,3M

        features = torch.stack((v, a, j, la), dim=-1) * plan_all_candidate_mask.unsqueeze(-1) # B,3M,4



        # ego_traj = torch.cat([ego_traj, yaw_angle.unsqueeze(-1), vec_scaler.unsqueeze(-1)],dim=-1) * plan_all_candidate_mask.unsqueeze(-1).unsqueeze(-1) # B, 3M, 50, 4

        '''
        delta_t = 0.1

        # 计算速度
        velocity = (ego_traj[:, :, 1:, :] - ego_traj[:, :, :-1, :]) / delta_t # B,3,49,2 单位m/s
        # 分离速度的 x 和 y 分量
        v_x = velocity[:, :, :, 0] # B,3,49
        v_y = velocity[:, :, :, 1] # B,3,49
        # 计算速度的大小
        v = torch.sqrt(v_x**2 + v_y**2) # # B,3,49

        # 计算加速度
        acceleration = (velocity[:, :, 1:, :] - velocity[:, :, :-1, :]) / delta_t # B,3,48,2
        # 分离加速度的 x 和 y 分量
        a_x = acceleration[:, :, :, 0] # B,3,48
        a_y = acceleration[:, :, :, 1] # B,3,48
        # 计算加速度的大小
        a = torch.sqrt(a_x**2 + a_y**2) # B,3,48

        # 计算加加速度
        jerk = (acceleration[:, :, 1:, :] - acceleration[:, :, :-1, :]) / delta_t # B,3,47,2
        # 分离加加速度的 x 和 y 分量
        j_x = jerk[:, :, :, 0] #B,3,47
        j_y = jerk[:, :, :, 1] #B,3,47
        # 计算加加速度的大小
        j = torch.sqrt(j_x**2 + j_y**2) #B,3,47

        # 计算曲率
        # 曲率的公式：κ = |v_x * a_y - v_y * a_x| / (v_x^2 + v_y^2)^(3/2)
        v_x_mid = v_x[:, :, :-1] # B,3,48
        v_y_mid = v_y[:, :, :-1] # B,3,48
        a_x_mid = a_x[:, :, :] # B,3,48
        a_y_mid = a_y[:, :, :] # B,3,48
        curvature = torch.abs(v_x_mid * a_y_mid - v_y_mid * a_x_mid) / ((v_x_mid**2 + v_y_mid**2)**1.5)

        # 计算横向加速度
        # 横向加速度的公式：a_lateral = v^2 * κ
        v_mid = v[:, :, :-1] # B,3,48
        lateral_acceleration = v_mid**2 * curvature # B,3,48

        # 计算航向角 (yaw angle)
        yaw_angle = torch.atan2(v_y, v_x)# B,3,49

        v_clone = v.clone()
        v = -v.mean(-1).clip(0, 15) / 15 # B,3,49 -> B,3
        a = a.abs().mean(-1).clip(0, 4) / 4 # B,3,48 -> B,3
        j = j.abs().mean(-1).clip(0, 6) / 6 # B,3,47 -> B,3
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5 # B,3,48 -> # B,3

        features = torch.stack((v, a, j, lateral_acceleration), dim=-1) # B,3,4

        v_clone = torch.cat([v_clone[:,:,:1],v_clone],dim=-1).unsqueeze(-1) # B,3,1 +B,3,49 = B,3,50,1
        yaw_angle = torch.cat([yaw_angle[:,:,:1],yaw_angle],dim=-1).unsqueeze(-1) # B,3,1 +B,3,49 = B,3,50,1

        ego_traj = torch.cat([ego_traj, yaw_angle, v_clone],dim=-1) # B, 3, 50, 4
        '''
        return features
        # return features, ego_traj
    
    def calculate_collision(self, ego_traj, agent_trajs, agent_trajs_prob, all_candidate_mask):
        # ego_traj: B, T, 4
        # agent_traj: B, N, 3m, T, 5
        # all_mask: B,N,3m

        # agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Compute the distance between the two agents
        B,N,M,T,_ = agent_trajs.shape
        dist = torch.norm(ego_traj[:, None, None, :, :2] - agent_trajs[:, :, :, :, :2], dim=-1) # B,N,3m,50,2 -> B,N,3m,50
        # dist = torch.ones(B,N,M,T).to(torch.float32).cuda()
        dist = dist.clamp(3,10)
        # Compute the collision cost
        # cost始终在[0,1]之间， dist小，cost越大，score就越小；
        cost = torch.exp(-0.2 * dist ** 2) * all_candidate_mask[:, :, :, None]# B,N,3m,50 * B,N,3m,1  
        cost = cost.sum(-1)*agent_trajs_prob# B,N,3m,50-> B,N,3m * B,N,3m
        cost = cost.sum(-1).sum(-1) # B,N,3m -> B
        
        # cost = torch.zeros(ego_traj.shape[0]).to(torch.float).cuda()
        return cost
    
    def get_latent_interaction_features(self, ego_traj, agent_traj, agent_traj_prob, agents_states, all_candidate_mask, agent_mask):
        '''
        之前是ego traj 和N个agent确定的【1】条轨迹计算交互值，现在是和每个agent的3m条计算交互值
        - ego_traj: B, 50,4 
        - agent_traj: B, N, 3m, 50, 5
        - agent_traj_prob: B,N,3m
        - agents_states: B, N, 13
        - ego_all_cand_mask: B

        - all_cand_mask: B,N,3m
        - agent_mask [B,N]
        输入一个场景：确定的ego 轨迹和多模态的agent轨迹，输出对这个场景的评分feature(4)
        '''

        # Get agent mask
        # agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Get relative attributes of agents

        relative_yaw = agent_traj[:, :, :, : , 2] - ego_traj[:, None, None, :, 2] # [B,N,3m,50]-[B,1,1,50] 计算一个ego的一个轨迹和N个agent的3M个轨迹在T时间区间内的相对角度
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw)) #[B,N,3m,50]
        relative_pos = agent_traj[:, :, :, :, :2] - ego_traj[:, None, None, :, :2] # [B,N,3m,50,2]-[B,1,1,50,2]
        relative_pos = torch.stack([relative_pos[..., 0] * torch.cos(relative_yaw), 
                                    relative_pos[..., 1] * torch.sin(relative_yaw)], dim=-1)# stack([B,N,3m,50],[B,N,3m,50])=[B,N,3m,50,2]

        agent_velocity = agent_traj[...,3:] #  B, N, 3m, 50, 2
        ego_velocity_x = ego_traj[:, :, 3] * torch.cos(ego_traj[:, :, 2]) # B,50 * B,50 先得到世界坐标下的x轴速度值，下面再转化为相对的x轴速度
        ego_velocity_y = ego_traj[:, :, 3] * torch.sin(ego_traj[:, :, 2]) # B,50 * B,50
        relative_velocity = torch.stack([(agent_velocity[..., 0] - ego_velocity_x[:, None,None,:]) * torch.cos(relative_yaw),# B,N,3m,50 - B,1,1,50
                                         (agent_velocity[..., 1] - ego_velocity_y[:, None,None,:]) * torch.sin(relative_yaw)], dim=-1) # stack->B,N,3m,50,2

        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1)# B,N,3m,50,5 =B,N,3m,50,2 + B,N,3m,50,1 + B,N,3m,50,2 得到1个ego的一个轨迹和N个agent的3M个轨迹在T区间的相对特征

        # Get agent attributes
        agent_attributes = agents_states[:, :, None,None, :].expand(-1, -1, relative_attributes.shape[2],relative_attributes.shape[3], -1) # B, N, 13-> B,N,1,1,13->B,N,3m,T,13
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1) # B,N,3m,50,5（相对）+13（agent）
        attributes = attributes * agent_mask[:, :, None, None, None] * all_candidate_mask[:,:,:,None,None]# 将mask掉的区域值置为0 # B,N,3m,50,18* B,N,1,1,1 * B,N,3M,1,1
        
        # mask->B,N,3M, 50, 18 ->encoder B,N,3M,50,256 -> prob -> B,N,3M,50, 256 ->max,max,mean-> B,256 ->decoder-> B,4
        B,N,M,T,D = attributes.shape
        features = self.interaction_feature_encoder(attributes) # B,N,3M,50,256
        features = features * (agent_traj_prob[:,:,:,None,None]**2) # B,N,3M,50, 256 * B,N,3M,1,1
        features = features.max(1).values.max(1).values.mean(1) # B,256
        features = self.interaction_feature_decoder(features) # B,4
        
        # task: B,N,3m,50,18 + B,N,3m  ->  B,4
        # self.decoder(attributes.mean(-2).reshape())
        


        # B,N,3m,18 
        # Encode relative attributes and decode to latent interaction features
        # features = self.interaction_feature_encoder(attributes) # B,N,3m,50,256      B,N,3m,512* prob -> BN3m,512
        # features = features.max(1).values.mean(2)# 该步融合概率！BN3m 256*prob?
        # features = self.interaction_feature_decoder(features)# B,3m,4
  
        return features


    def get_agent_yaw_vel(self,agents_traj,param, all_candidate_mask):
        '''
        B,N,3m,50,2 -> B,N,3m,50,5
        param:B,N,3M,(n_order+1),2
        all_candidate_mask:B,N,3m
        '''
        B,N,M,T,D = agents_traj.shape
        t_values = torch.linspace(0,1,T).cuda()

        vec_param = bezier_derivative(param) # B,N,3M,n,2
        vec_vectors = bezier_curve(vec_param, t_values)/5 # B,N,3M,50,2
        vec_vectors = vec_vectors.clamp(-15,15) * all_candidate_mask.unsqueeze(-1).unsqueeze(-1)
        # vec_scaler = torch.norm(vec_vectors, dim=-1)
        vx,vy = vec_vectors[...,0], vec_vectors[...,1]
        yaw_angle = torch.atan2(vy, vx) * all_candidate_mask.unsqueeze(-1)# B,N,3M,50
        yaw_angle = yaw_angle.unsqueeze(-1) # B,N,3M,50,1

    

        # delta_t = 0.1
        # # B,N,M,T,D = agents_traj.shape
        # # agents_traj += (torch.rand(B,N,M,T,D).cuda()-0.5)/10
        # # 计算速度
        # velocity = torch.diff(agents_traj, dim=-2)/delta_t
        # # velocity = (agents_traj[:, :, :, 1:, :] - agents_traj[:, :, :, :-1, :]) / delta_t # B,N,3m,49,2
        # velocity = velocity.clamp(-15,15)
        # # 分离速度的 x 和 y 分量
        # v_x = velocity[:, :, :, :, 0] # B,N,3m,49
        # v_y = velocity[:, :, :, :, 1] # B,N,3m,49
        # # 计算航向角 (yaw angle)
        # yaw_angle = torch.atan2(v_y, v_x)# B,N,3m,49
        # yaw_angle = torch.cat([yaw_angle[:, :, :, :1],yaw_angle],dim=-1).unsqueeze(-1) # B,N,3m,50,1


        # # agent_velocity = torch.diff(agents_traj[:, :, :, :2], dim=-2) / 0.1
        # velocity = torch.cat((velocity[:, :, :, :1, :], velocity), dim=-2) # B,N,3m,50,2



        agents_traj = torch.cat([agents_traj, yaw_angle, vec_vectors], dim=-1) # B,N,3m,50,2+1+2
        return agents_traj
        


    def forward(self, ego_trajs, plan_param, ego_encoding, agents_traj, param, agents_traj_probs, agents_states, all_candidate_mask, plan_all_candidate_mask, agent_mask, agent_vecs, agent_ctrs,mat_T):
        '''
        input:
            - ego_trajs: B,3M,50,2   
            - plan_params:  B,3M,(n_order+1)*2
            - ego_encoding: B,D128
            - agents_traj: B,N,3m,50,2
            - param: B,N,3m,(n_order+1)*2
            - agents_traj_probs: B,N,3m
            - agents_states: B,N,13        agent的特征编码（在obs点处）
            - all_candidate_mask: B,N,3m
            - plan_all_candidate_mask: B,3M
            - agent_mask: B,N

            - ctrs B,N,2
            - vecs B,N,2


            - enhanced ego_traj: B,3,50,4 加上了角度和速度标量 相对坐标系
            - enhanced agents_traj: B,N,3m,50,5 角度和速度（vx/vy）
        '''
        B,N,M,_,_= agents_traj.shape
        transforms = get_transform(agent_vecs) # B,N,2 -> B,N,2,2
        ego_trajs = transform_to_ori(ego_trajs, transforms[:,0,:,:], agent_ctrs[:,0,:], plan_all_candidate_mask) # B,3M,50,2 + B,2,2 + B,2
        agents_traj = transform_to_ori(agents_traj, transforms, agent_ctrs, all_candidate_mask) # B,N,3m,50,2 + B,N,2,2 + B,N,2
        plan_param = plan_param.reshape(B,plan_param.shape[1],self.n_order + 1,2) # B,3M,(n_order+1), 2
        ego_traj_features= self.get_hardcoded_features(ego_trajs, plan_param,plan_all_candidate_mask) #return B,3M,4vajc    B,3M,50,4 xy,yaw,vel

        plan_param = transform_to_ori(plan_param, transforms[:,0,:,:], agent_ctrs[:,0,:], plan_all_candidate_mask)# 绝对坐标系
        ego_trajs = self.get_ego_yaw_vel(ego_trajs, plan_param,plan_all_candidate_mask)
        param = param.reshape(B,N,M,self.n_order + 1,2) # B,N,3m,(n_order+1)*2
        param = transform_to_ori(param, transforms, agent_ctrs, all_candidate_mask)

        agents_traj = self.get_agent_yaw_vel(agents_traj,param,all_candidate_mask) # B,N,3m,50,5  xy,yaw,vel
        
        if not self._variable_cost:
            ego_encoding = torch.ones_like(ego_encoding)
        weights = self.weights_decoder(ego_encoding) # B,128->B,8  

        scores = []# B, 3M
        # cal interaction 
        for i in range(ego_trajs.shape[1]):# 3M
            hardcoded_features = ego_traj_features[:, i]# B,4             
            interaction_features = self.get_latent_interaction_features(ego_trajs[:, i], agents_traj, agents_traj_probs, agents_states, all_candidate_mask, agent_mask)# B,4
            features = torch.cat((hardcoded_features, interaction_features), dim=-1) # B,8
            score = -torch.sum(features * weights, dim=-1) # B,8 * B,8 -> B
            collision_feature = self.calculate_collision(ego_trajs[:, i], agents_traj, agents_traj_probs, all_candidate_mask)# B
            score += -10 * collision_feature # B
            scores.append(score)# 3M[B]

        scores = torch.stack(scores, dim=1) # B,3M
        scores = torch.where(plan_all_candidate_mask, scores, torch.tensor(float('-inf')).cuda())# True的地方为原值，False的地方为替换值

        return scores, weights # score越大代表越接近真人驾驶


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


def transform_to_ori(feat, transforms, ctrs, mask = None):
    '''
    feat: B,N,M,50,2
    feat: B,3M,50,2
    transforms: B,N,2,2
    ctrs: B,N,2
    mask : B,N,3m,
    mask : B,3m
    '''
    squeeze_flag = False
    if len(feat.shape) == 4:
        squeeze_flag = True
        feat = feat.unsqueeze(1) # B,1,3,50,2
        transforms = transforms.unsqueeze(1) # B,1,2,2
        ctrs = ctrs.unsqueeze(1) # B,1,2
    # B,N,M,50,2
    rot_inv = transforms.transpose(-2, -1) 
    feat[..., 0:2] = torch.matmul(feat[..., 0:2], rot_inv[:,:,None,:,:]) + ctrs[:,:,None,None,:] # B,N,M,50,2@B,N,1,2,2 + B,N,1,1,2
    if squeeze_flag:
        feat = feat.squeeze(1)
    if mask != None and len(mask.shape) == 3:
        # B,N,3m,
        B,N,M,T,D = feat.shape
        mask = (~mask.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,T,D)) # B,N,3m,T,D
        feat = feat.masked_fill(mask,0.0)
    elif mask != None and len(mask.shape) == 2:
        # B,3M
        B,M,T,D = feat.shape
        mask = (~mask.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,T,D)) # B,3M,T,D
        feat = feat.masked_fill(mask,0.0)

    return feat