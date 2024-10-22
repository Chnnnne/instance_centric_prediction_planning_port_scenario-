import os
import pickle
import numpy as np
import pandas as pd
import bisect
import itertools
import math
from pathlib import Path
from sklearn.model_selection import train_test_split

import multiprocessing

from map_point_seacher import MapPointSeacher
from modules.hdmap_lib.python.binding.libhdmap import HDMapManager, Vec2d

def judge_undefined_scene(x, y):
    a = -(80.0/77)
    b = 3715155.25974
    ans = y - a*x - b
    return True if ans <= 0 else False

    
def parse_log_data(log_data):
    data_info = {}
    ego_id = -1
    kDefaultCenterOffsetRatio = 0.401
    kDefaultBaseOffsetRatio = 0.295
    for i in range(len(log_data)):
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
            or math.hypot(agent_info['x'][index]-agent_info['x'][index+50], agent_info['y'][index]-agent_info['y'][index+50]) < 5:
                surr_ids.append((id_, index))
            else:
                target_ids.append((id_, index))
    return surr_ids, target_ids

def transform_to_local_coords(feat, center_xy, center_heading, heading_index=-1, type_index = -1):
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
        one_hot = np.eye(len(types))[np.searchsorted(types, feat[:, type_index])]
        feat = np.concatenate((feat[:, :-1], one_hot), axis=-1)
    return feat

def get_candidate_gt(candidate_points, gt_target):
    displacement = gt_target - candidate_points
    gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

    onehot = np.zeros(candidate_points.shape[0])
    onehot[gt_index] = 1

    offset_xy = gt_target - candidate_points[gt_index]
    return onehot, offset_xy

def pad_array_list(array_list):
    '''
    ayyay_list: 含有一系列二维矩阵，其中第一维的维度大小不一样
    '''
    # 找到最大的维度
    max_dim = max(arr.shape[0] for arr in array_list)

    # 填充数组为相同的维度并合并
    padded_array_list = [np.concatenate([arr, np.zeros((max_dim - arr.shape[0],) + arr.shape[1:])]) for arr in array_list]
    merged_array = np.stack(padded_array_list)
    return merged_array

def generate_future_feats(data_info: dict, target_ids: list):
    n = len(target_ids)
    tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask = [], [], [], [], []
    valid_flag = False
    for i in range(n):
        target_id, cur_index = target_ids[i]
        agent_info = data_info[target_id]
        center_xy = np.array([agent_info['x'][cur_index], agent_info['y'][cur_index]])
        center_heading = agent_info['vel_yaw'][cur_index]
        # 获取障碍物未来真实轨迹
        agt_traj_fut = np.column_stack((agent_info['x'][cur_index+1:cur_index+51].copy(), agent_info['y'][cur_index+1:cur_index+51].copy())).astype(np.float32)
        agt_traj_fut = transform_to_local_coords(agt_traj_fut, center_xy, center_heading)
        # 采样目标点
        ori = [agent_info['x'][cur_index], agent_info['y'][cur_index], 
               agent_info['vel'][cur_index], agent_info['vel_yaw'][cur_index], 
               agent_info['length'][cur_index], agent_info['width'][cur_index]]
        candidate_points = mp_seacher.get_candidate_target_points(ori)
        if len(candidate_points) == 0:
            candidate_points = np.zeros((1, 2))
            tar_candts_gt = np.zeros(1)
            tar_offset_gt = np.zeros(2)
            candts_mask = np.zeros((1))
        else:    
            candidate_points = np.asarray(candidate_points)
            candidate_points = transform_to_local_coords(candidate_points, center_xy, center_heading)
            tar_candts_gt, tar_offset_gt = get_candidate_gt(candidate_points, agt_traj_fut[-1, 0:2])
            if math.hypot(tar_offset_gt[0], tar_offset_gt[1]) > 2:
                candidate_points = np.zeros((1, 2))
                tar_candts_gt = np.zeros(1)
                tar_offset_gt = np.zeros(2)
                candts_mask = np.zeros((1))
            else:
                candts_mask = np.ones((candidate_points.shape[0]))
                valid_flag = True
        
        tar_candidate.append(candidate_points)
        gt_preds.append(agt_traj_fut)
        gt_candts.append(tar_candts_gt)
        gt_tar_offset.append(tar_offset_gt)
        candidate_mask.append(candts_mask)
            
    if not valid_flag:
        return None, None, None, None, None
    else:
        tar_candidate = pad_array_list(tar_candidate) # N, M, 2
        gt_preds = np.stack(gt_preds) # N, 50, 2
        gt_tar_offset = np.stack(gt_tar_offset) # N, 2
        gt_candts = pad_array_list(gt_candts)
        candidate_mask = pad_array_list(candidate_mask)
    return tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask

def generate_his_feats(data_info, agent_ids):
    agent_feats, agent_masks = [], []
    for agent_id, end_index in agent_ids:
        agent_feat = np.zeros((20, 7))
        agent_mask = np.zeros(20)
        start_index = end_index - 19
        index = 0
        if start_index < 0:
            start_index = 0
            index = abs(start_index)
        agent_info = data_info[agent_id]
        while start_index <= end_index:
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
    plan_traj = np.zeros((50, 4))
    plan_traj_mask = np.zeros(50)
    index = 0
    ego_info = data_info[-1]
    while index < 50:
        plan_index = ego_index + index + 1
        if plan_index >= len(ego_info['t']):
            break
        plan_traj[index] = np.array([ego_info['x'][plan_index], ego_info['y'][plan_index],
                                      ego_info['vel'][plan_index], ego_info['vel_yaw'][plan_index]])
        plan_traj_mask[index] = 1
        index += 1
    plan_feat, plan_mask = [], []
    for agent_id, index in target_ids:
        agent_info = data_info[agent_id]
        center_xy = np.array([agent_info['x'][index], agent_info['y'][index]])
        center_heading = agent_info['vel_yaw'][index]
        plan_traj_ = transform_to_local_coords(plan_traj.copy(), center_xy, center_heading, heading_index=3)
        plan_feat.append(plan_traj_)
        plan_mask.append(plan_traj_mask.copy())
    return np.stack(plan_feat), np.stack(plan_mask)

def pad_array(array, target_shape):
    padded_array = np.zeros(target_shape)
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
        lane_heading = lane.GetHeading(lane_s)
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
    d_pos = np.linalg.norm(ctrs[np.newaxis, :, :] - ctrs[:, np.newaxis, :], axis=-1)
    d_pos = d_pos * 2 / 100  # scale [0, radius] to [0, 2]
    pos_rpe = d_pos[np.newaxis, :]
    cos_a1 = get_cos(vecs[np.newaxis, :], vecs[:, np.newaxis])
    sin_a1 = get_sin(vecs[np.newaxis, :], vecs[:, np.newaxis])
    v_pos = ctrs[np.newaxis, :, :] - ctrs[:, np.newaxis, :] 
    cos_a2 = get_cos(vecs[np.newaxis, :], v_pos)
    sin_a2 = get_sin(vecs[np.newaxis, :], v_pos)

    ang_rpe = np.stack([cos_a1, sin_a1, cos_a2, sin_a2])
    rpe = np.concatenate([ang_rpe, pos_rpe], axis=0)
    rpe = np.transpose(rpe, (1, 2, 0))
    rpe_mask = np.ones((rpe.shape[0], rpe.shape[0]))
    return rpe, rpe_mask

def load_seq_save_features(index):
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
    for i in range(19, frame_num-50, 10):
        cur_t = ego_info['t'][i]
        # 过滤位于非有效地图上的数据
        if judge_undefined_scene(ego_info['x'][i], ego_info['y'][i]):
            continue
        
        # 获取当前帧周围的障碍物和需要预测的障碍物id
        surr_ids, target_ids = get_agent_ids(data_info, cur_t)
        if len(target_ids) == 0: 
            continue

        # 计算目标障碍物的目标点等特征
        tar_candidate, gt_preds, gt_candts, gt_tar_offset, candidate_mask = generate_future_feats(data_info, target_ids)
        if tar_candidate is None:
            continue
        # 计算障碍物的历史特征
        agent_ids = [(-1, i)]
        agent_ids.extend(target_ids)
        agent_ids.extend(surr_ids)
        agent_feats, agent_masks = generate_his_feats(data_info, agent_ids)
        agent_ctrs, agent_vecs = [], []
        for agent_id, index in agent_ids:
            agent_ctrs.append([data_info[agent_id]['x'][index], data_info[agent_id]['y'][index]])
            theta = data_info[agent_id]['vel_yaw'][index]
            agent_vecs.append([np.cos(theta), np.sin(theta)])
        agent_ctrs = np.asarray(agent_ctrs)
        agent_vecs = np.asarray(agent_vecs)

        # 计算plan特征
        plan_feat, plan_mask = generate_plan_feats(data_info, target_ids, i)

        # pad
        num = agent_feats.shape[0]
        pad_tar_candidate = pad_array(tar_candidate, (num, tar_candidate.shape[1], tar_candidate.shape[2])) # N, M, 2
        pad_gt_preds = pad_array(gt_preds, (num, gt_preds.shape[1], gt_preds.shape[2])) # N, 50, 2
        pad_gt_candts = pad_array(gt_candts, (num, gt_candts.shape[1])) # N, M
        pad_gt_tar_offset = pad_array(gt_tar_offset, (num, gt_tar_offset.shape[1])) # N, 2
        pad_candidate_mask = pad_array(candidate_mask,(num, candidate_mask.shape[1])) # N, M
        pad_plan_feat = pad_array(plan_feat, (num, plan_feat.shape[1], plan_feat.shape[2])) # N, 50, 4
        pad_plan_mask = pad_array(plan_mask, (num, plan_mask.shape[1])) # N, 50

        # 计算地图特征
        map_feats, map_mask, map_ctrs, map_vecs = generate_map_feats(data_info[-1], i, radius=80)
        if map_feats is None:
            continue

        # 计算rpe特征
        scene_ctrs = np.concatenate((agent_ctrs, map_ctrs), axis=0)
        scene_vecs = np.concatenate((agent_vecs, map_vecs), axis=0)
        rpe, rpe_mask = generate_rpe_feats(scene_ctrs, scene_vecs)

        feat_data = {}
        feat_data['agent_ctrs'] = agent_ctrs.astype(np.float32)
        feat_data['agent_vecs'] = agent_vecs.astype(np.float32)
        feat_data['agent_feats'] = agent_feats.astype(np.float32)
        feat_data['agent_mask'] = agent_masks.astype(np.int32)
        feat_data['tar_candidate'] = pad_tar_candidate.astype(np.float32)
        feat_data['candidate_mask'] = pad_candidate_mask.astype(np.int32)
        feat_data['gt_preds'] = pad_gt_preds.astype(np.float32)
        feat_data['gt_candts'] = pad_gt_candts.astype(np.float32)
        feat_data['gt_tar_offset'] = pad_gt_tar_offset.astype(np.float32)
        feat_data['plan_feat'] = pad_plan_feat.astype(np.float32)
        feat_data['plan_mask'] = pad_plan_mask.astype(np.int32)
        feat_data['map_ctrs'] = map_ctrs.astype(np.float32)
        feat_data['map_vecs'] = map_vecs.astype(np.float32)
        feat_data['map_feats'] = map_feats.astype(np.float32)
        feat_data['map_mask'] = map_mask.astype(np.int32)
        feat_data['rpe'] = rpe.astype(np.float32)
        feat_data['rpe_mask'] = rpe_mask.astype(np.int32)
        save_path = str(cur_output_path) + f'/{vehicle_name}_{cur_t}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(feat_data, f)
    return 

if __name__=="__main__": 
    map_file_path = "/fabupilot/release/resources/hdmap_lib/meishangang/map.bin"
    scene_type = 'port_meishan'
    HDMapManager.LoadMap(map_file_path, scene_type)
    hdmap = HDMapManager.GetHDMap()
    mp_seacher = MapPointSeacher(hdmap, t=5.0)
    
    input_path = '/private2/wanggang/pre_log_inter_data'
    all_file_list = [os.path.join(input_path, file) for file in os.listdir(input_path)]
    train_files, test_files = train_test_split(all_file_list, test_size=0.2, random_state=42)
    cur_files = test_files
    cur_output_path = '/private/wanggang/instance_centric_data/test'
    cur_output_path = Path(cur_output_path)
    if not cur_output_path.exists():
        cur_output_path.mkdir(parents=True)

    pool = multiprocessing.Pool(processes=8)
    pool.map(load_seq_save_features, range(len(cur_files)))
    print("###########完成###############")
    pool.close()
    pool.join()