import os,bisect,itertools,math,sys,pickle,time,multiprocessing
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from map_point_seacher import MapPointSeacher
from modules.hdmap_lib.python.binding.libhdmap import HDMapManager, Vec2d
from pathlib import Path
project_path = str(Path(__file__).resolve().parent.parent)
if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"add project_path:{project_path} to python search path")
import common.math_utils as math_utils
import common.plot_utils as plot_utils

from loguru import logger


def judge_undefined_scene(x, y):
    a = -(80.0/77)
    b = 3715155.25974
    ans = y - a*x - b
    return True if ans <= 0 else False

    
def parse_log_data(log_data):
    '''
    得到以id为key，坐标速度，长宽等信息为val的字典 data_info
    key = -1 代表ego
    '''
    data_info = {}
    ego_id = -1
    kDefaultCenterOffsetRatio = 0.401
    kDefaultBaseOffsetRatio = 0.295
    for i in range(len(log_data)): # 遍历这个以时间排序的，类型是world_debug数据的list
        cur_frame = log_data[i]
        cur_t = cur_frame.timestamp
        # 自车信息
        if ego_id not in data_info:
            data_info[ego_id] = {'t':[],'x':[], 'y':[], 'vel':[], 'vel_yaw':[], 'length':[], 'width':[], 'type':[]}
        data_info[ego_id]['t'].append(cur_t)
        data_info[ego_id]['x'].append(cur_frame.vehicle_state_debug.xy.x)
        data_info[ego_id]['y'].append(cur_frame.vehicle_state_debug.xy.y)
        data_info[ego_id]['vel'].append(cur_frame.vehicle_state_debug.vel)
        data_info[ego_id]['vel_yaw'].append(cur_frame.vehicle_state_debug.yaw)
        data_info[ego_id]['length'].append(6.855)
        data_info[ego_id]['width'].append(2.996)
        data_info[ego_id]['type'].append(-1)
        # 障碍物信息
        for agent in cur_frame.agent_map_debug.agents:
            # planning 内部agent_type编码方式
            # HUMAN = 0, CONTAINER_TRUCK = 1, TRUCK = 2, CAR = 3, BUS = 4, BICYCLE = 5, TRICYCLE = 6, CRANE = 7, GANTRY = 8,
            # BLOCK = 9, RAILING = 10, STACKER = 11, UNKNOWN = 12, GRID = 13, CHARGER = 14
            # container_truck和truck区分开
            if agent.agent_type not in [0, 1, 2, 3, 4, 11]:
                continue
            agent_id = agent.agent_id.id
            if agent_id not in data_info:
                data_info[agent_id] = {'t':[],'x':[], 'y':[], 'vel':[], 'vel_yaw':[], 'length':[], 'width':[], 'type':[]}
            if agent.HasField("connect_x"):
                x = agent.connect_x
                y = agent.connect_y
                if not agent.HasField("head_vel"):
                    vel = agent.vel
                else:
                    vel = agent.head_vel
                if not agent.HasField("head_vel_yaw"):
                    vel_yaw = agent.head_yaw
                else:
                    vel_yaw = agent.head_vel_yaw
                pos_yaw = agent.head_yaw
                length = agent.head_length
                width = agent.head_width
            elif agent.HasField("head_x"):
                # 基于挂车计算挂载点
                offset = agent.length * kDefaultCenterOffsetRatio 
                x = agent.x + offset * math.cos(agent.yaw)
                y = agent.y + offset * math.sin(agent.yaw)
                if not agent.HasField("head_vel"):
                    vel = agent.vel
                else:
                    vel = agent.head_vel
                if not agent.HasField("head_vel_yaw"):
                    vel_yaw = agent.head_yaw
                else:
                    vel_yaw = agent.head_vel_yaw
                pos_yaw = agent.head_yaw
                length = agent.head_length
                width = agent.head_width
            else:
                offset = 0.0 if agent.agent_type==0 else agent.length * kDefaultBaseOffsetRatio
                x = agent.x - offset*math.cos(agent.yaw)
                y = agent.y - offset*math.sin(agent.yaw)
                vel = agent.vel
                if not agent.HasField("vel_yaw"):
                    vel_yaw = agent.yaw
                else:
                    vel_yaw = agent.vel_yaw
                pos_yaw = agent.yaw
                length = agent.length
                width = agent.width
            if vel < 0.15:
                vel_yaw = pos_yaw
            data_info[agent_id]['t'].append(cur_t)
            data_info[agent_id]['x'].append(x)
            data_info[agent_id]['y'].append(y)
            data_info[agent_id]['vel'].append(vel)
            data_info[agent_id]['vel_yaw'].append(vel_yaw)
            data_info[agent_id]['length'].append(length)
            data_info[agent_id]['width'].append(width)
            agent_type = agent.agent_type  
            data_info[agent_id]['type'].append(agent_type)
    return data_info

def normalize_angle(angle):
    PI = math.pi
    return angle - 2*PI*np.floor((angle+PI)/(2*PI))   

def get_valid_index(agent_info, valid_t):
    index = -1
    left, right = 0, len(agent_info['t'])-1
    while left <= right:
        mid = (left+right)>>1
        if agent_info['t'][mid] > valid_t:
            right = mid - 1
        elif agent_info['t'][mid] < valid_t:
            left = mid + 1
        else:
            index = mid
            break
    return index


def get_agent_ids(data_info, cur_t):
    surr_ids, target_ids = list(), list()
    for id_, agent_info in data_info.items():
        if id_ == -1:
            continue
        index = get_valid_index(agent_info, cur_t)
        if index < 0:
            continue
        # 判断障碍物类型
        if agent_info['type'][index] == 0 or agent_info['vel'][index] < 0.15: # 行人或者静止障碍物
            surr_ids.append((id_, index))
        else:
            # 未来存在5s的真实轨迹
            if len(agent_info['t']) - index <= 50 \
            or math.hypot(agent_info['x'][index]-agent_info['x'][index+50], agent_info['y'][index]-agent_info['y'][index+50]) < 5 or\
                index < 10:
                surr_ids.append((id_, index))
            else:
                target_ids.append((id_, index))
    return surr_ids, target_ids

def transform_to_local_coords(feat, center_xy, center_heading, heading_index=-1, type_index = -1):
    '''
    以center_heading作为新坐标系的y轴
    feat 20,7
    # N, 20, 2
    # origin  N,2
    '''
    theta = math.pi/2 - center_heading
    rot = np.asarray([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]])
    feat[:, 0:2] = np.matmul((feat[:, 0:2] - center_xy), rot)

     # 转化角度
    if heading_index != -1:
        for i in range(len(feat)):
            feat[i, heading_index] = normalize_angle(feat[i, heading_index] - center_heading)

    # 转化类别
    if type_index != -1:
        # TODO（wg）后续移除VEHICLE、BUS
        #-1:ego, 0:'HUMAN', 1:CONTAINER_TRUCK, 2:TRUCK, 3:'CAR', 4:'BUS', 11:'STACKER'
        types = [-1, 0, 1, 2, 3, 4, 11] 
        one_hot = np.eye(len(types))[np.searchsorted(types, feat[:, type_index])] #  eye: 7*7     index: [20] -> [20, 7]
        feat = np.concatenate((feat[:, :-1], one_hot), axis=-1) # 20,6 + 7
    return feat

def get_candidate_gt(candidate_points, gt_target):
    displacement = gt_target - candidate_points
    gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

    onehot = np.zeros(candidate_points.shape[0])
    onehot[gt_index] = 1

    offset_xy = gt_target - candidate_points[gt_index]
    return onehot, offset_xy

def pad_array_list(array_list):
    '''# n, N,20, 2
    ayyay_list: 含有一系列二维矩阵，其中第一维的维度大小不一样
    '''
    # 找到最大的维度
    max_dim = max(arr.shape[0] for arr in array_list)

    # 填充数组为相同的维度并合并# n,
    padded_array_list = [np.concatenate([arr, np.zeros((max_dim - arr.shape[0],) + arr.shape[1:])])  # concate(arr:N, 20, 2 + zeros(remain,20,2)) = (max-N, 20,2)
                         for arr in array_list]
    merged_array = np.stack(padded_array_list) # n, Max-N,20,2
    return merged_array

def generate_future_feats(data_info: dict, target_ids: list):
    '''
        - tar_candidate = pad_array_list(tar_candidate) # n, Max-N, 2    n是target agent的个数， N（不定）是每个target的采样点的数量   Max-N是所有target agent分别采样点的数量的最大值
        - gt_preds = np.stack(gt_preds) # n, 50, 2
        - gt_tar_offset = np.stack(gt_tar_offset) # n, 2
        - gt_candts = pad_array_list(gt_candts) # n,Max-N
        - candidate_mask = pad_array_list(candidate_mask) # n,Max-N       标记是pad还是candidate  都已经转化为instance-centric
    '''
    n = len(target_ids)
    # candiadate points、 未来真实轨迹的末端点、与真实轨迹末端点最近的candiadate point标志位、与真实轨迹末端点最近的candiadate point加上offset得到真实轨迹的末端点、mask
    tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask = [], [], [], [], []
    valid_flag = False
    for i in range(n):
        target_id, cur_index = target_ids[i]
        agent_info = data_info[target_id]
        center_xy = np.array([agent_info['x'][cur_index], agent_info['y'][cur_index]])
        center_heading = agent_info['vel_yaw'][cur_index]
        # 获取障碍物未来真实轨迹 [50,2]
        agt_traj_fut = np.column_stack((agent_info['x'][cur_index+1:cur_index+51].copy(), agent_info['y'][cur_index+1:cur_index+51].copy())).astype(np.float32)
        agt_traj_fut = transform_to_local_coords(agt_traj_fut, center_xy, center_heading)
        # 采样目标点
        ori = [agent_info['x'][cur_index], agent_info['y'][cur_index], 
               agent_info['vel'][cur_index], agent_info['vel_yaw'][cur_index], 
               agent_info['length'][cur_index], agent_info['width'][cur_index]]
        candidate_points = mp_seacher.get_candidate_target_points(ori) # △ (N,2)
        if len(candidate_points) == 0:
            candidate_points = np.zeros((1, 2))
            tar_candts_gt = np.zeros(1)
            tar_offset_gt = np.zeros(2)
            candts_mask = np.zeros((1))
        else:    
            candidate_points = np.asarray(candidate_points)
            candidate_points = transform_to_local_coords(candidate_points, center_xy, center_heading)
            tar_candts_gt, tar_offset_gt = get_candidate_gt(candidate_points, agt_traj_fut[-1, 0:2]) # (N,) (2,)
            if math.hypot(tar_offset_gt[0], tar_offset_gt[1]) > 2:
                candidate_points = np.zeros((1, 2))
                tar_candts_gt = np.zeros(1)
                tar_offset_gt = np.zeros(2)
                candts_mask = np.zeros((1))
            else:
                candts_mask = np.ones((candidate_points.shape[0])) # (N, )
                valid_flag = True
        
        tar_candidate.append(candidate_points) # (n, N,2) n是target agent的个数， N（不定）是每个target的采样点的数量
        gt_preds.append(agt_traj_fut) # (n, 50, 2)
        gt_candts.append(tar_candts_gt) # (n, N,) 
        gt_tar_offset.append(tar_offset_gt) # (n, 2,)
        candidate_mask.append(candts_mask) # (n, N, )
            
    if not valid_flag:
        return None, None, None, None, None
    else:
        tar_candidate = pad_array_list(tar_candidate) # n, Max-N, 2   Max-N是所有target agent分别采样点的数量的最大值
        gt_preds = np.stack(gt_preds) # n, 50, 2
        gt_tar_offset = np.stack(gt_tar_offset) # n, 2
        gt_candts = pad_array_list(gt_candts) # n,Max-N
        candidate_mask = pad_array_list(candidate_mask) # n,Max-N       标记是pad还是candidate  都已经转化为instance-centric
    return tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask

def generate_ego_future_feats(ego_info: dict, cur_index: list):
    '''
        - ego_refpath_cords:(20, 2)  ndarray
        - ego_refpath_vecs: (20, 2)  ndarray
        - ego_vel_mode:  int
        - ego_gt_traj: (50,2)
    '''
    center_xy = np.array([ego_info['x'][cur_index], ego_info['y'][cur_index]])
    center_heading = ego_info['vel_yaw'][cur_index]
    ego_traj_fut_5s = np.column_stack((ego_info['x'][cur_index+1:cur_index+51].copy(), ego_info['y'][cur_index+1:cur_index+51].copy())).astype(np.float32)
    ego_traj_fut_5s = transform_to_local_coords(ego_traj_fut_5s, center_xy, center_heading)
    ego_traj_fut_15s = np.column_stack((ego_info['x'][cur_index+1:cur_index+151].copy(), ego_info['y'][cur_index+1:cur_index+151].copy())).astype(np.float32)
    distances = np.sqrt(np.sum(np.diff(ego_traj_fut_5s, axis=0)**2, axis=1))
    cumulative_distance = np.cumsum(distances)[-1]  # 真实轨迹在5s的累计距离
    ego_v = ego_info['vel'][cur_index]
    ego_vel_mode = 2 # 匀速
    if ego_v * 5 + 5 < cumulative_distance:
        ego_vel_mode = 1 # 加速
    elif ego_v * 5 - 5 > cumulative_distance:
        ego_vel_mode = 3 # 减速
            # 采样目标点
    ori = [ego_info['x'][cur_index], ego_info['y'][cur_index], 
            ego_info['vel'][cur_index], ego_info['vel_yaw'][cur_index], 
            ego_info['length'][cur_index], ego_info['width'][cur_index]]
    candidate_refpaths_cords, map_paths, candidate_refpaths_dis, kd_trees,keep_traj_idx = mp_seacher.get_candidate_refpath_and_sample_for_exact_dist_and_cluster_and_get_mappaths(ori)
    if len(candidate_refpaths_cords) == 0:
        return None, None, None, None, None, None
    else:
        gt_refpath_idx,_ = mp_seacher.get_candidate_gt_refpath_new(ego_traj_fut_15s,candidate_refpaths_cords, kd_trees)
        if gt_refpath_idx == -1:
            return None, None, None, None, None, None
        else:
            # keep_traj_idx 是经过聚类之后保存的traj的idx和mappath idx是对应的

            # 1. 此处利用mp_idx = keep_traj_idx[gt_idx]找到对应的filter之后的mappath 序号
            # 2. 然后获取这个序号的mappath末尾pathunit的lane，和周边lane ，拿到他们的id
            # 3. 遍历mappaths的每一个mappath的最后一个pu的lane id是否存在于neighbor ids中，
            # 4. 存在，则说明该mappath位于gt的周边车道， 记录这个mappath idx（也即是sample后的cords idx）到 neighbor_mppath_idces
            # 5. 如果这个idx仍在keep_traj——idx里，则完成了筛选且该refpath没有被聚类，将这个idx对应的cords加入neighbor cords中
            # 6. 合并gt ref cords和neighbor cords
            # gt = 0

            # 的mappath idx，再查一下keep_traj_idx是否是保存的idx，是的话，将其加入到ego的refpath
            gt_mappth_idx = keep_traj_idx[gt_refpath_idx]
            gt_mappath = map_paths[gt_mappth_idx]
            neighbor_lane_ids = []
            if gt_mappath[-1].lane.left_neighbor_forward_driving_lane() != None:
                neighbor_lane_ids.append(gt_mappath[-1].lane.left_neighbor_forward_driving_lane().id().value())
            if gt_mappath[-1].lane.right_neighbor_forward_driving_lane() != None:
                neighbor_lane_ids.append(gt_mappath[-1].lane.right_neighbor_forward_driving_lane().id().value())
            neighbor_mppath_idces = []
            for mp_idx, mp in enumerate(map_paths):
                if mp[-1].lane.id().value() in neighbor_lane_ids and mp_idx in keep_traj_idx:
                    neighbor_mppath_idces.append(mp_idx)
            neighbor_ref_cords = []
            for n_idx in neighbor_mppath_idces:
                neighbor_ref_cords.append(candidate_refpaths_cords[keep_traj_idx.index(n_idx)]) # append [50,2]


            all_ref_path = []
            all_ref_path.extend([candidate_refpaths_cords[gt_refpath_idx]]) # extend [[50,2]]
            all_ref_path.extend(neighbor_ref_cords) # extend [[50,2], [50,2]] len = M

            
            # plot_utils.draw_candidate_refpaths_with_his_fut(ori=ori,candidate_refpaths=candidate_refpaths_cords,cand_gt_idx=gt_refpath_idx,fut_traj=ego_traj_fut_15s)
            # plot_utils.draw_candidate_refpaths_with_his_fut(ori=ori,candidate_refpaths=all_ref_path,cand_gt_idx=0,fut_traj=ego_traj_fut_15s)
            # ego_refpath_cords = mp_seacher.sample_points(candidate_refpaths_cords[gt_refpath_idx], num=20, return_content="points")#(20,2)
            ego_refpath_cords = [mp_seacher.sample_points(ego_ref_cord, num=20,return_content="points") for ego_ref_cord in all_ref_path] # N[ndarray = (20,2) ...]   N=1,2,3
            # print(ego_refpath_cords)
            ego_refpath_vecs = mp_seacher.get_refpath_vec(ego_refpath_cords)
            ego_refpath_cords = [transform_to_local_coords(refpath_cord, center_xy, center_heading) for refpath_cord in ego_refpath_cords]
            ego_refpath_cords = np.asarray(ego_refpath_cords) # M,20,2    M=1,2,3
            ego_refpath_vecs = np.asarray(ego_refpath_vecs) # M,20,2    M=1,2,3
            ego_gt_cand = np.zeros(len(all_ref_path), dtype=np.int32) # M
            ego_gt_cand[0] = 1
            ego_vel_mode = np.asarray(ego_vel_mode) # 1
            ego_gt_traj = np.asarray(ego_traj_fut_5s) # 50,2
            ego_cand_mask = np.ones(len(all_ref_path), dtype=np.int32)
            return ego_refpath_cords, ego_refpath_vecs, ego_gt_cand, ego_vel_mode, ego_gt_traj, ego_cand_mask


    

def generate_future_feats_path(data_info: dict, target_ids: list):
    '''
        n是target agent的个数， N是每个agent采样的refpath个数， Max-N是所有agent最大的refpath个数
        all_candidate_refpaths_cords:   n , Max-N, 20,2       
        all_candidate_refpaths_vecs:    n , Max-N, 20,2     
        gt_preds:       n, 50, 2
        gt_cands:       n, Max-N, 
        gt_vel_mode:     n,       值只能是123，代表加、匀、减
        candidate_mask:      n, Max-N

    '''
    n = len(target_ids)
    all_candidate_refpaths_cords, all_candidate_refpaths_vecs, gt_preds, all_gt_candts, gt_vel_mode, all_candidate_mask = [], [], [], [], [], []
    valid_flag = False
    for i in range(n):
        target_id, cur_index = target_ids[i]
        agent_info = data_info[target_id]
        center_xy = np.array([agent_info['x'][cur_index], agent_info['y'][cur_index]])
        center_heading = agent_info['vel_yaw'][cur_index]
        # 获取障碍物未来真实轨迹 [50,2]
        before_idx = max(cur_index - 19, 0)
        agt_traj_his = np.column_stack((agent_info['x'][before_idx:cur_index].copy(), agent_info['y'][before_idx:cur_index].copy())).astype(np.float32)
        agt_traj_fut = np.column_stack((agent_info['x'][cur_index+1:cur_index+51].copy(), agent_info['y'][cur_index+1:cur_index+51].copy())).astype(np.float32)
        agt_traj_fut = transform_to_local_coords(agt_traj_fut, center_xy, center_heading)
        agt_traj_fut_all = np.column_stack((agent_info['x'][cur_index+1:].copy(), agent_info['y'][cur_index+1:].copy())).astype(np.float32)
        # 计算该agent未来5s的加减速行为
        distances = np.sqrt(np.sum(np.diff(agt_traj_fut, axis=0)**2, axis=1))
        cumulative_distance = np.cumsum(distances)[-1]  # 真实轨迹在5s的累计距离
        obs_v = agent_info['vel'][cur_index]
        vel_mode = 2
        if obs_v * 5 + 5 < cumulative_distance:
            vel_mode = 1 # 加速
        elif obs_v * 5 - 5 > cumulative_distance:
            vel_mode = 3 # 减速
        # 采样目标点
        ori = [agent_info['x'][cur_index], agent_info['y'][cur_index], 
               agent_info['vel'][cur_index], agent_info['vel_yaw'][cur_index], 
               agent_info['length'][cur_index], agent_info['width'][cur_index]]
        # candidate_refpaths_cord, candidate_refpaths_vec, map_paths = mp_seacher.get_candidate_refpaths(ori) # △  (N, max_l)
        candidate_refpaths_cords, map_paths, candidate_refpaths_dis, kd_trees, keep_traj_idx = mp_seacher.get_candidate_refpath_and_sample_for_exact_dist_and_cluster_and_get_mappaths(ori)
        valid_flag = False
        if len(candidate_refpaths_cords) == 0:
            gt_cand = np.zeros(1) #(1,)
            candidate_refpaths_cords = np.zeros((1, 20, 2)) # (1,20,2)
            candidate_refpaths_vecs = np.zeros((1, 20, 2))#(1,20,2)
            candts_mask = np.zeros((1))#(1, )
        else:   
            gt_idx,_ = mp_seacher.get_candidate_gt_refpath_new(agt_traj_fut_all,candidate_refpaths_cords, kd_trees)
            if gt_idx == -1:
                gt_cand = np.zeros(1) #(1,)
                candidate_refpaths_cords = np.zeros((1, 20, 2)) # (1,20,2)
                candidate_refpaths_vecs = np.zeros((1, 20, 2))#(1,20,2)
                candts_mask = np.zeros((1))#(1, )
            else:
                # plot_utils.draw_candidate_refpaths_with_his_fut(ori=ori,candidate_refpaths=candidate_refpaths_cords,cand_gt_idx=gt_idx,his_traj=agt_traj_his,fut_traj=agt_traj_fut_all)
                gt_cand = np.zeros(len(candidate_refpaths_cords),dtype=np.int32) # (N,)
                gt_cand[gt_idx] = 1
                candidate_refpaths_cords = [mp_seacher.sample_points(refpath_cords, num=20,return_content="points") for refpath_cords in candidate_refpaths_cords] # list[ndarray:shape(50,2)] ->  (N, 20, 2)
                candidate_refpaths_vecs = mp_seacher.get_refpath_vec(candidate_refpaths_cords) #list[ndarray] (N, 20, 2)
                candidate_refpaths_vecs = np.asarray(candidate_refpaths_vecs)# (N, 20, 2)
                candidate_refpaths_cords = [transform_to_local_coords(refpath_cord, center_xy, center_heading) for refpath_cord in candidate_refpaths_cords]
                candidate_refpaths_cords = np.asarray(candidate_refpaths_cords) # (N, 20, 2)
                candts_mask = np.ones((candidate_refpaths_cords.shape[0])) # (N,)
                valid_flag = True

        all_candidate_refpaths_cords.append(candidate_refpaths_cords) #(n, N, 20, 2)
        all_candidate_refpaths_vecs.append(candidate_refpaths_vecs)# (n, N, 20, 2)
        gt_preds.append(agt_traj_fut) # n, 50, 2
        gt_vel_mode.append(vel_mode) # n
        all_gt_candts.append(gt_cand) # n, N
        all_candidate_mask.append(candts_mask) # n ,N
            
    if not valid_flag:# 如果改taget_ids列表都是无效的数据就直接返回None
        return None, None, None, None, None, None
    else:
        all_candidate_refpaths_cords = pad_array_list(all_candidate_refpaths_cords) # n, Max-N,20, 2
        all_candidate_refpaths_vecs = pad_array_list(all_candidate_refpaths_vecs) # n, Max-N, 20, 2
        gt_preds = np.stack(gt_preds) # n, 50, 2
        gt_vel_mode = np.array(gt_vel_mode) # n,
        all_gt_candts = pad_array_list(all_gt_candts) # n,Max-N
        all_candidate_mask = pad_array_list(all_candidate_mask) # n,Max-N       标记是pad还是candidate  都已经转化为instance-centric
    return all_candidate_refpaths_cords, all_candidate_refpaths_vecs, gt_preds, gt_vel_mode, all_gt_candts, all_candidate_mask

def generate_his_feats(data_info, agent_ids):
    '''

        - return: 所有agent，包含ego和surr agent、target agent的  instance-centric视角下的信息。 all_n = num of all agent
            - agent_feats: [all_n, 20, 13]
            - agent_mask: [all_n,20]
    '''
    agent_feats, agent_masks = [], []
    for agent_id, end_index in agent_ids:# agent
        agent_feat = np.zeros((20, 7))
        agent_mask = np.zeros(20)
        start_index = end_index - 19
        index = 0
        if start_index < 0:
            start_index = 0
            index = abs(start_index)
        agent_info = data_info[agent_id]
        while start_index <= end_index: # index
            agent_feat[index] = np.array([agent_info['x'][start_index], agent_info['y'][start_index],
                                          agent_info['vel'][start_index], agent_info['vel_yaw'][start_index],
                                          agent_info['length'][start_index], agent_info['width'][start_index],
                                          agent_info['type'][start_index]])
            agent_mask[index] = 1
            start_index += 1
            index += 1
        center_xy = np.array([agent_info['x'][end_index], agent_info['y'][end_index]])
        center_heading = agent_info['vel_yaw'][end_index]
        agent_feat = transform_to_local_coords(agent_feat, center_xy, center_heading, heading_index=3, type_index=6)
        agent_feats.append(agent_feat)
        agent_masks.append(agent_mask)
    return np.stack(agent_feats), np.stack(agent_masks)

def generate_plan_feats(data_info, target_ids, ego_index):
    '''
        在tagret-centric视角也即每个agent（包括ego）的不同视角下，自车的未来50帧信息， n = target num
        - plan_feat:   [all_n, 50, 4]   
        - plan_mask:   [all_n, 50]
    '''
    plan_traj = np.zeros((50, 4))
    plan_traj_mask = np.zeros(50)
    index = 0
    ego_info = data_info[-1]
    while index < 50: # plan_traj:自车的未来50帧轨迹 (50, 4)
        plan_index = ego_index + index + 1
        if plan_index >= len(ego_info['t']):
            break
        plan_traj[index] = np.array([ego_info['x'][plan_index], ego_info['y'][plan_index],
                                      ego_info['vel'][plan_index], ego_info['vel_yaw'][plan_index]]) # (50,4)
        plan_traj_mask[index] = 1
        index += 1
    plan_feat, plan_mask = [], []
    for agent_id, index in target_ids:
        agent_info = data_info[agent_id]
        center_xy = np.array([agent_info['x'][index], agent_info['y'][index]]) # ego_index帧对应的target agent的位置和朝向
        center_heading = agent_info['vel_yaw'][index]
        plan_traj_ = transform_to_local_coords(plan_traj.copy(), center_xy, center_heading, heading_index=3)
        plan_feat.append(plan_traj_) # [all_n, 50, 4]
        plan_mask.append(plan_traj_mask.copy()) # [all_n, 50]
    return np.stack(plan_feat), np.stack(plan_mask)

def pad_array(array, target_shape):
    '''
    从1开始有可能会溢出？？  应该不会？因为all_n必定比n大，因为ego的存在
    arr=n , Max-N, 20,2            target_shape=all_n, Max-N, 20, 2
    '''
    padded_array = np.zeros(target_shape)# 
    padded_array[1:1+array.shape[0]] = array
    return padded_array

def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

def get_lane_infos(lanes, center_point, center_heading, distance=10, radius=100, num_points_each_polyline=20):
    types_map = {"junction":0, "lane":1}
    lane_polylines, lane_polylines_mask = [], []
    lane_ctrs, lane_vecs = [], []
    center_xy = np.array([center_point.x(), center_point.y()])
    for lane in lanes:
        if lane.IsInJunction():
            continue
        lane_s, _ = lane.GetProjection(center_point)
        lane_heading = lane.GetHeading(lane_s) # ego观察点投影到lane的lane heading
        if lane.bi_direction_lane():
            diff_angle = normalize_angle(center_heading - lane_heading)
            if abs(diff_angle) > math.pi/2:
                continue
        reference_line = lane.reference_line()
        length = lane.length()
        s = 0
        count = 0
        polyline = []
        while s < length:
            if count >= num_points_each_polyline:
                break
            point = reference_line.GetReferencePoint(s)
            point = Vec2d(point.x(), point.y())
            if (point - center_point).Length() < radius:
                count += 1
                polyline.append([point.x(), point.y(), types_map["lane"]])
            s += distance
        if count == 0:
            continue
        polyline = np.asarray(polyline)
        lane_ctr = np.mean(polyline[:, 0:2], axis=0)
        lane_vec = [np.cos(lane_heading), np.sin(lane_heading)]
        polyline_dir = get_polyline_dir(polyline[:, 0:2].copy())
        polyline = transform_to_local_coords(polyline, lane_ctr, lane_heading)
        polyline = np.concatenate((polyline[:, 0:2], polyline_dir, polyline[:, 2:]), axis=-1)
        valid_num, point_dim =  min(num_points_each_polyline, polyline.shape[0]), polyline.shape[-1]
        cur_polyline = np.zeros((num_points_each_polyline, point_dim))
        cur_polyline_mask = np.zeros((num_points_each_polyline))
        cur_polyline[:valid_num] = polyline[:valid_num]
        cur_polyline_mask[:valid_num] = 1
        lane_polylines.append(cur_polyline)
        lane_polylines_mask.append(cur_polyline_mask)
        lane_ctrs.append(lane_ctr)
        lane_vecs.append(lane_vec)
    if len(lane_polylines)==0:
        return None, None, None, None
    lane_polylines = np.stack(lane_polylines)
    lane_polylines_mask = np.stack(lane_polylines_mask)
    lane_ctrs = np.stack(lane_ctrs)
    lane_vecs= np.stack(lane_vecs)
    return lane_polylines, lane_polylines_mask, lane_ctrs, lane_vecs        

def get_junction_infos(junctions, distance=5.0, num_points_each_polyline=20):
    types_map = {"junction":0, "lane":1}
    junction_polylines, junction_polylines_mask = [], []
    junction_ctrs, junction_vecs = [], []
    for junction in junctions:
        # 过滤船头船尾设置的虚拟路口
        if junction.is_virtual_junction():
            if "vessel_head_and_tail" in junction.attributes().attributes().values():
                continue
        points= junction.polygon().points()
        # my_utils.draw_points(points=points)
        points.append(points[0])
        s = [0] + [math.hypot(points[i].x() - points[i-1].x(), points[i].y() - points[i-1].y()) for i in range(1, len(points))]
        s = list(itertools.accumulate(s))

        polyline = []
        cur_length = 0
        count = 0
        while cur_length < s[-1]:
            if count >= num_points_each_polyline:
                break
            s_idx = bisect.bisect_left(s, cur_length)
            t = (cur_length - s[s_idx-1]) / (s[s_idx] - s[s_idx-1])
            x = (1-t)*points[s_idx-1].x() + t*points[s_idx].x()
            y = (1-t)*points[s_idx-1].y() + t*points[s_idx].y()
            polyline.append([x, y, types_map["junction"]])
            cur_length += distance
            count += 1
        polyline = np.asarray(polyline)
        junction_ctr = np.mean(polyline[:, 0:2], axis=0)
        junction_vec = [np.cos(math.pi/2), np.sin(math.pi/2)]
        polyline_dir = get_polyline_dir(polyline[:, 0:2].copy())
        polyline = transform_to_local_coords(polyline, junction_ctr, math.pi/2)
        polyline = np.concatenate((polyline[:, 0:2], polyline_dir, polyline[:, 2:]), axis=-1)

        valid_num, point_dim = min(num_points_each_polyline, polyline.shape[0]), polyline.shape[-1]
        cur_polyline = np.zeros((num_points_each_polyline, point_dim))
        cur_polyline_mask = np.zeros((num_points_each_polyline))
        cur_polyline[:valid_num] = polyline[:valid_num]
        cur_polyline_mask[:valid_num] = 1
        junction_polylines.append(cur_polyline)
        junction_polylines_mask.append(cur_polyline_mask)
        junction_ctrs.append(junction_ctr)
        junction_vecs.append(junction_vec)
    if len(junction_polylines)==0:
        return None, None, None, None
    junction_polylines = np.stack(junction_polylines)
    junction_polylines_mask = np.stack(junction_polylines_mask)
    junction_ctrs = np.stack(junction_ctrs)
    junction_vecs = np.stack(junction_vecs)
    return junction_polylines, junction_polylines_mask, junction_ctrs, junction_vecs
    
def generate_map_feats(ego_info, index, radius = 70):
    '''
    - map_feats     map_element_num, 20, 5
    - map_mask      map_element_num, 20
    - map_ctrs      map_element_num, 2
    - map_vecs      map_element_num, 2
    '''
    center_point = Vec2d(ego_info["x"][index], ego_info["y"][index])
    center_heading = ego_info["vel_yaw"][index]
    lanes = hdmap.GetLanes(center_point, radius)
    junctions = hdmap.GetJunctions(center_point, radius)
    lane_polylines, lane_polylines_mask, lane_ctrs, lane_vecs = get_lane_infos(lanes, center_point, center_heading)
    
    junction_polylines, junction_polylines_mask, junction_ctrs, junction_vecs = get_junction_infos(junctions)
    if lane_polylines is None and junction_polylines is None:
        return None, None, None, None
    elif lane_polylines is None:
        return junction_polylines, junction_polylines_mask, junction_ctrs, junction_vecs
    elif junction_polylines is None:
        return lane_polylines, lane_polylines_mask, lane_ctrs, lane_vecs
    else:
        map_polylines = np.concatenate((lane_polylines, junction_polylines), axis=0)
        map_polylines_mask = np.concatenate((lane_polylines_mask, junction_polylines_mask), axis=0)
        map_ctrs = np.concatenate((lane_ctrs, junction_ctrs), axis=0)
        map_vecs = np.concatenate((lane_vecs, junction_vecs), axis=0)
        return map_polylines, map_polylines_mask, map_ctrs, map_vecs

def get_cos(v1, v2):
    ''' 输入: [M, N, 2], [M, N, 2]
        输出: [M, N]
        cos(<a,b>) = (a·b) / |a||b|
    '''
    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
    return cos_dang

def get_sin(v1, v2):
    ''' 输入: [M, N, 2], [M, N, 2]
        输出: [M, N]
        sin(<a,b>) = (a×b) / |a||b|
    '''
    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
    return sin_dang
    
def generate_rpe_feats(ctrs, vecs):
    '''
    ctrs: all,2
    vecs: all,2
    '''
    d_pos = np.linalg.norm(ctrs[np.newaxis, :, :] - ctrs[:, np.newaxis, :], axis=-1) # 1,all,2 - all, 1, 2 -> all,all,2 -> all,all
    d_pos = d_pos * 2 / 100  # scale [0, radius] to [0, 2]
    pos_rpe = d_pos[np.newaxis, :] # 1,all,all,2
    cos_a1 = get_cos(vecs[np.newaxis, :], vecs[:, np.newaxis]) # 1,all,2     all,1,2
    sin_a1 = get_sin(vecs[np.newaxis, :], vecs[:, np.newaxis]) # 
    v_pos = ctrs[np.newaxis, :, :] - ctrs[:, np.newaxis, :]  # 1,all,2 - all, 1, 2 = all,all,2
    cos_a2 = get_cos(vecs[np.newaxis, :], v_pos) # 1,all,2  
    sin_a2 = get_sin(vecs[np.newaxis, :], v_pos) # 

    ang_rpe = np.stack([cos_a1, sin_a1, cos_a2, sin_a2])
    rpe = np.concatenate([ang_rpe, pos_rpe], axis=0)# 5,all,all
    rpe = np.transpose(rpe, (1, 2, 0))# all,all,5
    rpe_mask = np.ones((rpe.shape[0], rpe.shape[0]))
    return rpe, rpe_mask


def my_candidate_refpath_search_test(index):
    pickle_path = cur_files[index]
    print(pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    log_data = data['data']
    
    cur_x, cur_y = log_data[0].vehicle_state_debug.xy.x, log_data[0].vehicle_state_debug.xy.y
    last_x, last_y = log_data[-1].vehicle_state_debug.xy.x, log_data[-1].vehicle_state_debug.xy.y
    # 过滤位于非有效地图上的数据
    if judge_undefined_scene(cur_x, cur_y) or judge_undefined_scene(last_x, last_y):
        return
    data_info = parse_log_data(log_data) # data_info: ego/agent's info,  key:id  val:info
    ego_info = data_info[-1] # ego_info连续性保证？
    frame_num = len(ego_info['t'])
    vehicle_name = pickle_path.split('/')[-1].split('_')[0]
    for i in range(19, frame_num-50, 10): # 10f间隔遍历ego的所有f obs:2s fut:5s
        if i <=1609:
            continue
        cur_t = ego_info['t'][i]
        # 获取当前帧周围的障碍物和需要预测的障碍物id
        surr_ids, target_ids = get_agent_ids(data_info, cur_t)
        if len(target_ids) == 0:
            continue
        n = len(target_ids)
        for t in range(n):
            print("="*100,"new agent start")
            target_id, cur_index = target_ids[t]
            agent_info = data_info[target_id]
            center_xy = np.array([agent_info['x'][cur_index], agent_info['y'][cur_index]])
            center_heading = agent_info['vel_yaw'][cur_index]
            # agt_traj_fut_5s = np.column_stack((agent_info['x'][cur_index+1:cur_index+51].copy(), 
            #                                 agent_info['y'][cur_index+1:cur_index+51].copy())).astype(np.float64)
            agt_traj_fut_all = np.column_stack((agent_info['x'][cur_index+1:].copy(), 
                                            agent_info['y'][cur_index+1:].copy())).astype(np.float64) 
            before_index = max(cur_index - 19,0)
            agt_traj_his_2s = np.column_stack((agent_info['x'][before_index:cur_index+1].copy(), 
                                            agent_info['y'][before_index:cur_index+1].copy())).astype(np.float64)
            ori = [agent_info['x'][cur_index], agent_info['y'][cur_index], 
               agent_info['vel'][cur_index], agent_info['vel_yaw'][cur_index], 
               agent_info['length'][cur_index], agent_info['width'][cur_index]]

            # candidate_refpaths_cord, candidate_refpaths_vec, map_paths = mp_seacher.get_candidate_refpaths(ori)
            candidate_refpaths_cords, map_paths, candidate_refpaths_dis, kd_trees = mp_seacher.get_candidate_refpath_and_sample_for_exact_dist_and_cluster_and_get_mappaths(ori)
            cand_gt_idx,errors = mp_seacher.get_candidate_gt_refpath_new(agt_traj_fut_all,candidate_refpaths_cords, kd_trees)
            
            if cand_gt_idx == -1:
                print("filter!")
                print("!!"*100)
                
            # return gt_idx
            # cand_gt_idx,way = mp_seacher.get_candidate_gt_refpath(agt_traj_fut_all, map_paths, candidate_refpaths_cords)
            # if cand_gt_idx == -1:
            #     print("filter！")
            #     cand_gt_idx = None
                
            plot_utils.draw_candidate_refpaths_with_his_fut(ori,candidate_refpaths_cords, cand_gt_idx, agt_traj_his_2s ,agt_traj_fut_all, other_info = {"pkl_path":pickle_path, "pkl_index":index, "ego_frame":i, "way": " ","target_id":target_id, "errors":errors})
            print("="*100,"new agent end")
            
def check_nan_inf_abnormal(tensor_dict, other_info):
    flag = False
    for key in tensor_dict.keys():
        if np.isinf(tensor_dict[key]).any():
            logger.debug(f"{key} contains inf\n")
            flag = True
        if np.isnan(tensor_dict[key]).any():
            logger.debug(f"{key} contains nan\n")
            flag = True
        if (tensor_dict[key] > 2000).any():
            logger.debug(f"{key} bigger than 2000, val:{tensor_dict[key]}\n")
            flag = True
    if flag:
        logger.debug(other_info)
    return flag


def load_seq_save_features(index):
    '''
    对每个pkl: cur_files[idx]单独提取
    '''
    pickle_path = cur_files[index]
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    log_data = data['data']
    data_info = parse_log_data(log_data)
    if data_info is None:
        return
    ego_info = data_info[-1]
    frame_num = len(ego_info['t'])
    vehicle_name = pickle_path.split('/')[-1].split('_')[0]
    for i in range(19, frame_num-160, 20): # 10f间隔遍历ego的所有f obs:2s fut:5s
        cur_t = ego_info['t'][i]
        # 过滤位于非有效地图上的数据
        if judge_undefined_scene(ego_info['x'][i], ego_info['y'][i]):
            continue
        
        # 获取当前帧周围的障碍物和需要预测的障碍物id
        surr_ids, target_ids = get_agent_ids(data_info, cur_t)
        if len(target_ids) == 0: 
            continue

        # 计算目标障碍物的目标点等特征
        # tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask = generate_future_feats(data_info, target_ids)
        candidate_refpaths_cords, candidate_refpaths_vecs, gt_preds, gt_vel_mode, gt_candts, candidate_mask = generate_future_feats_path(data_info, target_ids)

        ego_refpath_cords, ego_refpath_vecs, ego_gt_cand, ego_vel_mode, ego_gt_traj, ego_cand_mask = generate_ego_future_feats(ego_info, i)


        if candidate_refpaths_cords is None or ego_refpath_cords is None:
            continue
        # 计算障碍物的历史特征
        agent_ids = [(-1, i)]
        agent_ids.extend(target_ids)
        agent_ids.extend(surr_ids)
        agent_feats, agent_masks = generate_his_feats(data_info, agent_ids)

                    
        agent_ctrs, agent_vecs = [], [] # 存储所有agent的obs点信息 (all_n,2) (all_n,2)
        for agent_id, index in agent_ids:
            agent_ctrs.append([data_info[agent_id]['x'][index], data_info[agent_id]['y'][index]])
            theta = data_info[agent_id]['vel_yaw'][index]
            agent_vecs.append([np.cos(theta), np.sin(theta)])
        agent_ctrs = np.asarray(agent_ctrs)
        agent_vecs = np.asarray(agent_vecs)

        # 计算plan特征
        plan_feat, plan_mask = generate_plan_feats(data_info, target_ids, i) 

        # pad
        num = agent_feats.shape[0]# num of all_n
        pad_candidate_refpaths_cords = pad_array(candidate_refpaths_cords, (num, candidate_refpaths_cords.shape[1], candidate_refpaths_cords.shape[2], candidate_refpaths_cords.shape[3])) # all_n, max-N, 20,2
        pad_candidate_refpaths_vecs = pad_array(candidate_refpaths_vecs, (num, candidate_refpaths_vecs.shape[1], candidate_refpaths_vecs.shape[2], candidate_refpaths_vecs.shape[3])) # all_n, max-N, 20,2
        pad_gt_preds = pad_array(gt_preds, (num, gt_preds.shape[1], gt_preds.shape[2])) # all_n, 50, 2
        pad_gt_vel_mode = pad_array(gt_vel_mode, (num))# all_n, 
        pad_gt_candts = pad_array(gt_candts, (num, gt_candts.shape[1])) # all_n, max-N
        pad_candidate_mask = pad_array(candidate_mask,(num, candidate_mask.shape[1])) # all_n, max-N
        pad_plan_feat = pad_array(plan_feat, (num, plan_feat.shape[1], plan_feat.shape[2])) # all_n, 50, 4
        pad_plan_mask = pad_array(plan_mask, (num, plan_mask.shape[1])) # all_n, 50

        # 计算地图特征
        map_feats, map_mask, map_ctrs, map_vecs = generate_map_feats(data_info[-1], i, radius=80)
        if map_feats is None:
            continue

        # 计算rpe特征
        scene_ctrs = np.concatenate((agent_ctrs, map_ctrs), axis=0) # n,2    map_n,2
        scene_vecs = np.concatenate((agent_vecs, map_vecs), axis=0) # n,2    map_n,2
        rpe, rpe_mask = generate_rpe_feats(scene_ctrs, scene_vecs)

        feat_data = {}
        feat_data['agent_ctrs'] = agent_ctrs.astype(np.float32) # (all_n,2) 
        feat_data['agent_vecs'] = agent_vecs.astype(np.float32) # (all_n,2)
        feat_data['agent_feats'] = agent_feats.astype(np.float32) # [all_n, 20, 13]
        feat_data['agent_mask'] = agent_masks.astype(np.int32) # [all_n, 20]
        # mask pos to zero
        feat_data['agent_feats'][~feat_data['agent_mask'].astype(bool)] = 0

        

        feat_data['ego_refpath_cords'] = ego_refpath_cords.astype(np.float32) # (M,20,2)
        feat_data['ego_refpath_vecs'] = ego_refpath_vecs.astype(np.float32) # (M,20,2)
        feat_data['ego_vel_mode'] = ego_vel_mode.astype(np.int32) # (1, )
        feat_data['ego_gt_cand'] = ego_gt_cand.astype(np.float32) # (M, )
        feat_data['ego_gt_traj'] = ego_gt_traj.astype(np.float32) #(50,2)
        feat_data['ego_cand_mask'] = ego_cand_mask.astype(np.int32) # (M,)

        feat_data['candidate_refpaths_cords'] = pad_candidate_refpaths_cords.astype(np.float32)# all_n, max-N, 20,2
        feat_data['candidate_refpaths_vecs'] = pad_candidate_refpaths_vecs.astype(np.float32)# all_n, max-N, 20,2
        feat_data['gt_preds'] = pad_gt_preds.astype(np.float32)# all_n, 50,2
        feat_data['gt_vel_mode'] = pad_gt_vel_mode.astype(np.int32) # all_n
        feat_data['gt_candts'] = pad_gt_candts.astype(np.float32) # all_n, max-N
        feat_data['candidate_mask'] = pad_candidate_mask.astype(np.int32) # all_n, max-N
        # mask pos to zero
        feat_data['candidate_refpaths_cords'][~feat_data['candidate_mask'].astype(bool)] = 0
        feat_data['candidate_refpaths_vecs'][~feat_data['candidate_mask'].astype(bool)] = 0

        # feat_data['tar_candidate'] = pad_tar_candidate.astype(np.float32)
        # feat_data['candidate_mask'] = pad_candidate_mask.astype(np.int32)
        # feat_data['gt_preds'] = pad_gt_preds.astype(np.float32)
        # feat_data['gt_candts'] = pad_gt_candts.astype(np.float32)
        # feat_data['gt_tar_offset'] = pad_gt_tar_offset.astype(np.float32)
        feat_data['plan_feat'] = pad_plan_feat.astype(np.float32) # [all_n, 50, 4]   
        feat_data['plan_mask'] = pad_plan_mask.astype(np.int32)  # [all_n, 50]
        feat_data['map_ctrs'] = map_ctrs.astype(np.float32) # map_element_num, 2
        feat_data['map_vecs'] = map_vecs.astype(np.float32) # map_element_num, 2
        feat_data['map_feats'] = map_feats.astype(np.float32) # map_element_num, 20, 5
        feat_data['map_mask'] = map_mask.astype(np.int32) # map_element_num, 20
        # mask pos to zero
        feat_data['map_feats'][~feat_data['map_mask'].astype(bool)] = 0




        feat_data['rpe'] = rpe.astype(np.float32)
        feat_data['rpe_mask'] = rpe_mask.astype(np.int32)
        save_path = str(cur_output_path) + f'/{vehicle_name}_{cur_t}.pkl'
        check_dict = {"agent_feats":feat_data['agent_feats'], "agent_vecs":feat_data['agent_vecs'], 
                      "ego_refpath_cords":feat_data['ego_refpath_cords'], "ego_refpath_vecs": feat_data['ego_refpath_vecs'],"ego_gt_traj":feat_data['ego_gt_traj'],
                    "candidate_refpaths_cords": feat_data['candidate_refpaths_cords'], "candidate_refpaths_vecs":feat_data['candidate_refpaths_vecs'],
                    "gt_preds":feat_data['gt_preds'], "map_feats":feat_data['map_feats']
        }
        if check_nan_inf_abnormal(check_dict, other_info = f"pickle_path:{pickle_path}, i:{i}"):
            continue
        with open(save_path, 'wb') as f:
            pickle.dump(feat_data, f)
            print(f"file_index:{index}, i:{i}, pkl saved at{save_path}")
            print("$"*80)

    return

if __name__=="__main__": 
    
    logger.add("runtime{time}.log")

    logger.debug("debug begin")
    map_file_path = "/fabupilot/release/resources/hdmap_lib/meishangang/map.bin"
    scene_type = 'port_meishan'
    HDMapManager.LoadMap(map_file_path, scene_type)
    hdmap = HDMapManager.GetHDMap()
    mp_seacher = MapPointSeacher(hdmap, t=5.0)
    
    # input_path = '/private2/wanggang/pre_log_inter_data'
    input_path = '/private/wangchen/instance_model/pre_log_inter_data_small'
    all_file_list = [os.path.join(input_path, file) for file in os.listdir(input_path)]
    # all_file_list = all_file_list[:int(len(all_file_list)/10)]
    all_file_list = all_file_list[:3]
    train_files, test_files = train_test_split(all_file_list, test_size=0.2, random_state=42)
    cur_files = all_file_list
    print(f"共需处理{len(cur_files)}个pkl")# 1w+
    
    # cur_output_path = '/private/wangchen/instance_model/instance_model_data_small/train'
    cur_output_path = '/private/wangchen/instance_model/instance_model_data_test_data_generate_latency/'
    cur_output_path = Path(cur_output_path)
    if not cur_output_path.exists():
        cur_output_path.mkdir(parents=True)
    start_time = time.time()
    pool = multiprocessing.Pool(processes=16)
    pool.map(load_seq_save_features, range(len(cur_files)))

    # for i in range(10,100): # 19 error 21 draw
    #     print("--"*20, i)
    # #     # my_candidate_refpath_search_test(i)
    #     load_seq_save_features(i)

    # print("###########完成###############")
    pool.close()
    pool.join()
    # 结束计时
    end_time = time.time()

    # 计算并打印耗时
    total_time = end_time - start_time
    print("耗时",total_time)
    






    # import math, os, sys
    # from pathlib import Path
    # project_path = str(Path(__file__).resolve().parent.parent)
    # if project_path not in sys.path:
    #     sys.path.append(project_path)
    #     print(f"add project_path:{project_path} to python search path")
    # import common.math_utils as math_utils
    # import common.plot_utils as plot_utils
    # import common.map_utils as map_utils
    # hdmap = map_utils.get_hdmap()
    # mp_seacher = MapPointSeacher(hdmap, t=5.0)
    # input_path = '/private/wangchen/pre_log_inter_data_small'
    # all_file_list = [os.path.join(input_path, file) for file in os.listdir(input_path)]
    # train_files, test_files = train_test_split(all_file_list, test_size=0.2, random_state=42)
    # cur_files = train_files
    # cur_output_path = '/private/wangchen/instance_model_data_small/train'
    # cur_output_path = Path(cur_output_path)
    # if not cur_output_path.exists():
    #     cur_output_path.mkdir(parents=True)
    # print("start")
    # for i in range(len(cur_files)): # 19 error 21 draw
    #     print("--"*20, i)
    #     # my_candidate_refpath_search_test(i)
    #     load_seq_save_features(i)
    #     if i >= 28:
    #         break
    # print("complete")
    
