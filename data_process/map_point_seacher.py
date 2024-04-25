import math, os, sys
from pathlib import Path
project_path = str(Path(__file__).resolve().parent.parent)
if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"add project_path:{project_path} to python search path")
import numpy as np
from modules.hdmap_lib.python.binding.libhdmap import Vec2d
import common.math_utils as math_utils
import common.plot_utils as plot_utils
import common.map_utils as map_utils

kSearchJunctionRadius = 2.0
kJunctionLength = 0.0
kMinPathLength = 10.0

class PathUnit():
    def __init__(self, lane=None, junction=None, start_s=None, end_s=None, is_reverse=None):
        self.lane = lane
        self.junction = junction
        self.start_s = start_s
        self.end_s = end_s
        self.is_reverse = is_reverse
    
    def is_junction(self, ):
        if self.junction:
            return True
        return False
    
    def length(self, ):
        if self.start_s != None:
            return abs(self.start_s - self.end_s)
        return None
    
    def __hash__(self):
        if self.junction:
            return hash(self.junction.id().value())
        return hash((self.lane.id().value(), self.start_s, self.is_reverse))

    def __eq__(self, other):
        if self.junction and other.junction:
            return self.junction.id().value() == other.junction.id().value()
        elif self.lane and other.lane:
            return self.lane.id().value() == other.lane.id().value() and self.is_reverse == other.is_reverse and int(self.start_s) == int(other.start_s)
        else:
            return False
    
    def print_info(self, ):
        if self.junction:
            print("junction: ", self.junction.id().value())
        else:
            print("lane: {}, start_s: {}, end_s: {}, is_reverse: {}".format(self.lane.id().value(), self.start_s, self.end_s, self.is_reverse))
            

class MapPointSeacher():
    def __init__(self, hdmap, t=5.0, sample_distance=1.0):
        self.hdmap = hdmap
        self.t = t
        self.distance = sample_distance
        self.length = 6
        self.width = 2.7
        self.search_range = None
        self.agent_point = None
        self.agent_front_point = None
        self.agent_yaw = None
    
    def get_candidate_target_points_base_grid(self, ori):
        # ori: [x, y, vel, yaw, length, width]
        self.agent_point = Vec2d(ori[0], ori[1])
        self.agent_yaw = ori[3]
        lon_vec = Vec2d(self.distance*math.cos(self.agent_yaw), self.distance*math.sin(self.agent_yaw))
        lat_vec = Vec2d(-self.distance*math.sin(self.agent_yaw), self.distance*math.cos(self.agent_yaw))
        candidate_points = []
        for i in range(15):
            for j in range(20):
                lat_pt = 7-i
                lon_pt = 1+j
                x = self.agent_point.x() + lat_pt*lat_vec.x() + lon_pt*lon_vec.x()
                y = self.agent_point.y() + lat_pt*lat_vec.y() + lon_pt*lon_vec.y()
                candidate_points.append([x, y])
        return candidate_points
    
    
    def get_candidate_target_points(self, ori):
        # ori: [x, y, vel, yaw]
        search_radius = 2.7
        self.agent_point = Vec2d(ori[0], ori[1])
        self.agent_yaw = ori[3]
        self.length = min(ori[4], 17.0)
        self.width = min(ori[5], 4)
        self.agent_front_point = Vec2d(ori[0] + 0.795*self.length*math.cos(self.agent_yaw), ori[1] +  0.795*self.length*math.sin(self.agent_yaw))
        start_pathunits = []
        candidate_points = []
        if self.get_start_pathunits(start_pathunits):
            map_paths = []
            self.search_range = self.get_search_range(ori[2])
            for i in range(len(start_pathunits)):
                start_pathunit = start_pathunits[i]
                map_path = []
                searched_length = 0.0
                self.dfs(start_pathunit, searched_length, map_path, map_paths)

            # grid_candidate_points = self.get_candidate_target_points_base_grid(ori)
            # candidate_points = self.sample_candidate_points_v2(map_paths, grid_candidate_points)
            print("sample_candidate_points map path len", len(map_paths))
            candidate_points = self.sample_candidate_points(map_paths)
            # print(f"总共有{len(map_paths)}条map_path")
            # for map_path in map_paths:
            #     print("##"*12)
            #     for pathunit in map_path:
            #         pathunit.print_info()

            if len(candidate_points) == 0:
                candidate_points = self.get_candidate_target_points_base_grid(ori)
                print("inner candidate_points = 0, now excute base grid")

        if len(candidate_points) == 0:
            candidate_points = self.get_candidate_target_points_base_grid(ori)
            print("outer candidate_points = 0, now excute base grid")

            return candidate_points
        return candidate_points
    
    def get_candidate_refpaths(self, ori):
        '''
            根据ori信息，得到refpaths，也即每个agent有若干个refpath，每个refpath由若干个lane seg组成
            return [N, max_lane_seg, 2]      mask[N, max_lane_seg]
        '''
        # ori: [x, y, vel, yaw]
        search_radius = 2.7
        self.agent_point = Vec2d(ori[0], ori[1])
        self.agent_yaw = ori[3]
        self.length = min(ori[4], 17.0)
        self.width = min(ori[5], 4)
        self.agent_front_point = Vec2d(ori[0] + 0.795*self.length*math.cos(self.agent_yaw), ori[1] +  0.795*self.length*math.sin(self.agent_yaw))
        start_pathunits = []
        candidate_refpaths_cord = []
        if self.get_start_pathunits(start_pathunits):
            map_paths = []
            self.search_range = self.get_search_range_refpath(ori[2])
            for i in range(len(start_pathunits)):
                start_pathunit = start_pathunits[i]
                map_path = []
                searched_length = 0.0
                self.dfs(start_pathunit, searched_length, map_path, map_paths)
            candidate_refpaths_cord, candidate_refpaths_vec = self.sample_candidate_refpaths(map_paths, ori)# N条refpath, max lane segments

        if len(candidate_refpaths_cord) == 0:
            # candidate_refpaths = self.get_candidate_target_points_base_grid(ori)
            print("error candidate_refpaths = 0")
            print("map_paths:",map_paths)
            exit()
        else:
            # self.sample_candidate_refpaths_for_mo
            pass
            
            # candidate_refpaths = self.get_candidate_refpaths_points_base_grid(ori)
            
        # candidate_refpaths = filter_candidate_refpaths(candidate_refpaths)
        return candidate_refpaths_cord, candidate_refpaths_vec, map_paths

    def get_candidate_gt_refpath(self, agt_traj_fut_all, candidate_mappaths, candidate_refpaths_cord):
        agt_gt_traj_mappath = get_mappath_from_traj(agt_traj_fut_all)
        gt_cand = []
        for mappath in candidate_mappaths:
            if mappath in agt_gt_traj_mappath:
                gt_cand.append(mappath)
        if len(gt_cand) == 1:
            pass # 真实轨迹比较长
        elif len(gt_cand) > 1:
            pass # 不存在？
        elif  len(gt_cand) == 0:
            if agt_gt_traj_mappath[-1].is_junction():
                pass # 基于长度路径 等插值采样选出最小的那一个
            else:
                pass # 真实轨迹太短了，过滤该case


    def sample_candidate_refpaths(self, map_paths, ori):
        '''
            (从该agent位于的ori的位置朝向和地图等信息)对dfs得到的map_paths，得到一系列的map_path得到refpaths，  
            一个map path（若干个lane和juncton pathunit组成）对应一个refpath

        '''
        refpaths_cord = []
        refpaths_vec = []
        unique_pre_lane = set()
        for mappath in map_paths:
            refpath = []
            for idx in range(len(mappath)):
                refpath.extend(self.sample_pathunit(mappath, idx, refpath, ori))
            
            refpath = np.asarray(refpath, np.float64)
            refpath_cord = (refpath[:-1,0:2] + refpath[1:,0:2])/2
            diff_vec = refpath[1:,0:2] - refpath[:-1,0:2]
            diff_vec_norm = np.linalg.norm(diff_vec, axis = 1)
            diff_vec_norm = np.stack((diff_vec_norm,diff_vec_norm), axis=1)
            refpath_unit_vec = (diff_vec)/diff_vec_norm 
            np.linalg.norm(refpath[1:,0:2] - refpath[:-1,0:2], axis = 1)[:,np.newaxis]
            refpaths_cord.append(refpath_cord)
            refpaths_vec.append(refpath_unit_vec)
        return refpaths_cord,refpaths_vec # list[ num_point,2]
            

        

    def sample_pathunit(self, mappath, idx, refpath, ori):
        '''
        return 该pathunit对应的lane segs
        '''
        pathunit = mappath[idx]
        point = Vec2d()
        lane_segs = []
        junctions_dict = {}
        if pathunit.is_junction():

            # print("lane-junction-lane or junction-lane mode")
            in_point = [self.agent_point.x(), self.agent_point.y()]
            in_heading = self.agent_yaw
            if idx > 0:
                in_pathunit = mappath[idx - 1]
                in_pathunit.lane.GetPoint(in_pathunit.end_s, 0.0, point)
                in_point = refpath[-1] if len(refpath) !=0 else [point.x(),point.y()]
                in_heading = in_pathunit.lane.GetHeading(in_pathunit.end_s)
                if in_pathunit.is_reverse:
                    in_heading += math.pi
            if idx + 1 < len(mappath):
                out_pathunit = mappath[idx+1]
                out_pathunit.lane.GetPoint(out_pathunit.start_s, 0.0, point)
                out_point = [point.x(), point.y()]
                out_heading = out_pathunit.lane.GetHeading(out_pathunit.start_s)
                if out_pathunit.is_reverse:
                    out_heading += math.pi
                curve_bez,_ = math_utils.get_bezier_function(in_point, in_heading, out_point, out_heading)
                # print("add junction point")
                # print(curve_bez.evaluate_multi(np.linspace(0, 1, 50)).T.shape)
                split_num = math.ceil(math_utils.get_bezier_curve_length(curve_bez) + 1)
                junction_points = curve_bez.evaluate_multi(np.linspace(0, 1, split_num)).T
                junction_points = junction_points[1:-1] if len(junction_points) >= 2 else []
                lane_segs.extend(junction_points)
            else:
                print("该mappath没有离开路口的后继lane")

        else:
            s = pathunit.start_s
            if pathunit.is_reverse:
                while s > pathunit.end_s:
                    pathunit.lane.GetPoint(s, 0.0, point)
                    dist = (point - self.agent_point).Length()
                    s -= self.distance
                    if dist > self.search_range[1]:
                        continue
                    elif dist < self.search_range[0]:
                        continue
                    if not self.is_valid_candidate_point(point): 
                        break
                    lane_segs.append([point.x(), point.y()])
            else:
                while s < pathunit.end_s:
                    pathunit.lane.GetPoint(s, 0.0, point)
                    dist = (point - self.agent_point).Length()
                    s += self.distance
                    if dist > self.search_range[1]:
                        continue
                    elif dist < self.search_range[0]:
                        continue
                    if not self.is_valid_candidate_point(point): 
                        continue
                    lane_segs.append([point.x(), point.y()])

        return lane_segs

    def sample_candidate_points_v2(self, map_paths, grid_candidate_points):
        point = Vec2d()
        unique_pathunits = set()
        junctions = []
        candidate_points = []
        for map_path in map_paths:
            finish_flag = False
            for pathunit in map_path:
                if pathunit in unique_pathunits:
                    continue
                unique_pathunits.add(pathunit)
                if pathunit.is_junction():
                    junctions.append(pathunit.junction)
                    continue
                s = pathunit.start_s
                if not pathunit.is_reverse:
                    while s < pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        dist = (point - self.agent_point).Length()
                        s += self.distance
                        if dist > self.search_range[1]:
                            finish_flag = True
                            break
                        elif dist < self.search_range[0]:
                            continue
                        if not self.is_valid_candidate_point(point):
                            break
                        candidate_points.append([point.x(), point.y()])
                else:
                    while s > pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        dist = (point - self.agent_point).Length()
                        s -= self.distance
                        if dist > self.search_range[1]:
                            finish_flag = True
                            break
                        elif dist < self.search_range[0]:
                            continue
                        if not self.is_valid_candidate_point(point):
                            break
                        candidate_points.append([point.x(), point.y()]) 
                if finish_flag:
                    break
        if len(junctions) != 0:
            for point in grid_candidate_points:
                dist = (Vec2d(point[0], point[1]) - self.agent_point).Length()     
                if dist > self.search_range[1] or dist < self.search_range[0]:
                    continue
                for junction in junctions:
                    if junction.polygon().IsPointIn(Vec2d(point[0], point[1])):
                        candidate_points.append(point)
                        break
        return candidate_points
                   
    def get_start_pathunits(self, start_pathunits):
        '''
        - 前提：用ori作为搜索条件。 
        - 返回：结果存储在start_pathunits，并完善每个pathunit的reverse信息和start_s信息（junction pathunit无需完善）
        - 逻辑：
            - 以车头前侧位置作为起始搜索点
                - 先搜索Junction，只要是非船头船尾的虚拟路口都加入start_pathunits
                - 再搜索lane
                    - 如果不在路口里且非双向车道，直接进一般set_reverse_s逻辑(根据agent和车道的相对角度信息，得到pathunits并设置start_s和reverse并加入start_pathunits)
                    - 如果是双向车道则disregard其中的角度差大于90度的一条，只考虑另外的一条加入start_pathunits(进set_reverse_s)
                    
            - start_pathunits存在非反向且非路口的path时,  is_filter_reverse = True
            - 以车头前侧位置作为起始搜索点,车宽为半径搜索（已经加入进来Junction和lane的不做考虑）
                - 搜索lane，同
                    - #如果之前小搜索范围时已经搜索到了非逆向行驶的pathunit，这里再搜索到逆向行驶的pathunit就不考虑了
                - 搜索Junction，同


        - PathUnit
            PathUnit存储的是车道或者路口对象,
            1. 还有agent处于该车道或者路口中的相对距离信息(start_s和end_s)
            2. agent进入该车道时相对车道的起始距离(s)信息

            
            当agent逆行时，next pathunit的start_s是前继link的from_s,本pathunit的end_s是前继link的to_s

            final:agent进入该pathunit时位于车道的位置就是start_s，离开时是end_s,逆行时 start_s > end_s
        '''
        unique_pathunit_ids = set()
        search_yaw = self.agent_yaw
        junctions = self.hdmap.GetJunctions(self.agent_front_point, 0.1)
        for junction in junctions:
            if junction.id().value() not in unique_pathunit_ids:
                unique_pathunit_ids.add(junction.id().value())
                # 过滤船头船尾设置的虚拟路口
                if junction.is_virtual_junction():
                    if "vessel_head_and_tail" in junction.attributes().attributes().values():
                        continue
                pathunit = PathUnit(junction=junction)
                start_pathunits.append(pathunit)
                break
        lanes = self.hdmap.GetLanes(self.agent_front_point, 0.1)
        for lane in lanes:
            unique_pathunit_ids.add(lane.id().value())
            if not lane.IsInJunction() and not lane.bi_direction_lane(): # lane不在交叉口中 且 非双向车道
                pathunit = PathUnit(lane=lane)
                pathunits = self.set_reverse_and_s(pathunit)
                start_pathunits.extend(pathunits)
                break
            elif lane.bi_direction_lane(): # 双向车道
                lane_s, _ = lane.GetProjection(self.agent_point)
                lane_heading = lane.GetHeading(lane_s)
                diff_angle = self.normalize_angle(search_yaw - lane_heading)
                if abs(diff_angle) > math.pi/2:
                    continue
                else: #[0, pi/2]
                    pathunit = PathUnit(lane=lane)
                    pathunits = self.set_reverse_and_s(pathunit)
                    start_pathunits.extend(pathunits)
                    break
        is_filter_reverse = False # start_pathunits存在非反向且非路口的path时,  is_filter_reverse = True
        for pathunit in start_pathunits:
            if not pathunit.is_junction() and not pathunit.is_reverse:
                is_filter_reverse = True
                break
        lanes = self.hdmap.GetLanes(self.agent_front_point, self.width) #扩大搜索范围
        junctions = self.hdmap.GetJunctions(self.agent_front_point, self.width)
        for lane in lanes:
            if lane.id().value() in unique_pathunit_ids:
                continue
            unique_pathunit_ids.add(lane.id().value())
            if not lane.IsInJunction() and not lane.bi_direction_lane():
                pathunit = PathUnit(lane=lane)
                pathunits = self.set_reverse_and_s(pathunit)
                if is_filter_reverse and len(pathunits) == 1 and pathunits[0].is_reverse:#如果之前小搜索范围时已经搜索到了非逆向行驶的pathunit，这里再搜索到逆向行驶的pathunit就不考虑了
                    continue
                start_pathunits.extend(pathunits)
            elif lane.bi_direction_lane():
                lane_s, _ = lane.GetProjection(self.agent_point)
                lane_heading = lane.GetHeading(lane_s)
                diff_angle = self.normalize_angle(search_yaw - lane_heading)
                if abs(diff_angle) > math.pi/2:
                    continue
                else:
                    pathunit = PathUnit(lane=lane)
                    pathunits = self.set_reverse_and_s(pathunit)
                    if is_filter_reverse and len(pathunits) == 1 and pathunits[0].is_reverse:
                        continue
                    start_pathunits.extend(pathunits)
                             
        for junction in junctions:
            if junction.id().value() not in unique_pathunit_ids:
                unique_pathunit_ids.add(junction.id().value())
                 # 过滤船头船尾设置的虚拟路口
                if junction.is_virtual_junction():
                    if "vessel_head_and_tail" in junction.attributes().attributes().values():
                        continue
                pathunit = PathUnit(junction=junction)
                start_pathunits.append(pathunit)          
        return len(start_pathunits) > 0
    
    def get_search_range(self, vel):
        a = 2 #m/s^2
        min_search_length = max(vel*self.t - 25.0, 1.0)
        max_search_length = max(vel*self.t + 25.0, min_search_length)
        max_search_length = min(max_search_length, 80)
        return [min_search_length, max_search_length]
    
    def get_search_range_refpath(self, vel):
        min_search_length = max(vel/10, 0.2)
        max_search_length = vel*self.t + 25.0
        return [min_search_length, max_search_length]
    
    def set_reverse_and_s(self, pathunit):
        '''
        - desc: 
            根据agent的ori信息和车道的相对角度信息，计算agent在该pathunit的start_s和是否逆行的的信息，返回设置后的pathunits

        - detail:
            如果lane在agent点的方向与agent_yaw的角度差
            在[0, pi/3]为正常区间，正常行驶
            在[pi/3, pi*2/3]，则agent在该lane上可能为逆行，也可能不逆行，因此加入两个pathunit
            在[pi*2/3, pi]记为agent在该pathunit对应的lane上为逆行
        '''
        if pathunit.is_junction():
            return [pathunit]
        is_reverse = False
        # 障碍物xy坐标转化成sd坐标
        s,_ = pathunit.lane.GetProjection(self.agent_point)
        lane_heading = pathunit.lane.GetHeading(s)
        pathunit.start_s = s
        diff_angle = abs(self.normalize_angle(lane_heading - self.agent_yaw))
        # 双向都要搜索
        if diff_angle > math.pi/3 and diff_angle < 2*math.pi/3:#[pi/3, pi*2/3]
            reverse_pathunit = PathUnit()
            reverse_pathunit.lane = pathunit.lane
            reverse_pathunit.start_s = pathunit.start_s # ？应是lane.length-start_s
            reverse_pathunit.is_reverse = True
            pathunit.is_reverse = False
            return [pathunit, reverse_pathunit]
        elif diff_angle <= math.pi/3:#[0, pi/3]
            pathunit.is_reverse = False
            return [pathunit]
        else:# [pi*2/3, pi]
            pathunit.is_reverse= True
            return [pathunit]
    
    # 将角度转化到[-pi,pi)之间
    def normalize_angle(self, angle):
        PI = math.pi
        return angle - 2*PI*np.floor((angle+PI)/(2*PI))
    
    def copy_map_path(self, map_path):
        map_path_copy = []
        for pathunit in map_path:
            copy_pathunit = PathUnit(lane=pathunit.lane, junction=pathunit.junction, start_s=pathunit.start_s, end_s=pathunit.end_s, is_reverse=pathunit.is_reverse)
            map_path_copy.append(copy_pathunit)
        return map_path_copy
    
    def dfs(self, pathunit, searched_length, map_path, map_paths):  
        '''
            
            pathunit:当前搜索位置
            searched_length:进入该函数时已经搜索的距离
            map_path：当前搜索的路径pathunit的list
            已知：pathunit-lane有starts和reverse       pathunit-junction无starts和reverse.
            本函数需要完善starts、ends、reverse   （get start units要完善starts和reverse）
        '''
        if searched_length > self.search_range[1]:
            map_path_copy = self.copy_map_path(map_path)
            map_paths.append(map_path_copy)
            return 
        
        next_pathunits = []
        pathunit_end_ss = []
        
        if pathunit.is_junction():
            # 初始位置在路口中
            if len(map_path) == 0:
                next_pathunits = self.get_driving_path_of_junction(pathunit.junction)
            else:
                next_pathunits = self.get_exit_path_of_junction(pathunit.junction, map_path[-1])
        # 位于非路口
        else:
            is_reverse = pathunit.is_reverse
            # 障碍物逆行，则获取前继
            if is_reverse:
                most_likely_lane_link = self.get_predecessor_lane(pathunit.lane, pathunit.start_s)
            # 障碍物正向行驶，则获取后继
            else:
                most_likely_lane_link = self.get_successor_lane(pathunit.lane, pathunit.start_s)

            # （pathunit非路口）没有获取到前后继,获取后续路口
            if most_likely_lane_link == None:
                lane_end_point = Vec2d()
                lane_s = 0.0 if is_reverse else pathunit.lane.length()
                pathunit.lane.GetPoint(lane_s, 0.0, lane_end_point)# 获取pathunit的末端点然后获取junction
                # 搜索当前点附近(2m)的junction
                possible_junctions = self.hdmap.GetJunctions(lane_end_point, kSearchJunctionRadius)
                end_s = min(lane_s, pathunit.start_s) if is_reverse else max(lane_s, pathunit.start_s)
             
                for possible_junction in possible_junctions:
                    # 过滤船头船尾设置的虚拟路口
                    if possible_junction.is_virtual_junction():
                        if "vessel_head_and_tail" in possible_junction.attributes().attributes().values():
                            continue
                    next_pathunit = PathUnit(junction=possible_junction)
                    next_pathunits.append(next_pathunit)
                    pathunit_end_ss.append(end_s)
                    break
                if len(pathunit_end_ss) == 0:
                    pathunit.end_s = end_s
            else: #  成功获取了前后继
                if is_reverse: # 获取了前继
                    next_pathunit = PathUnit()
                    lane = most_likely_lane_link.from_lane()
                    next_pathunit.lane = lane
                    next_pathunit.start_s = most_likely_lane_link.from_s() 
                    next_pathunit.is_reverse = is_reverse
                    pathunit_end_ss.append(most_likely_lane_link.to_s())
                    next_pathunits.append(next_pathunit)
                else: # 获取了后继
                    next_pathunit = PathUnit()
                    lane = most_likely_lane_link.to_lane()
                    next_pathunit.lane = lane
                    next_pathunit.start_s = most_likely_lane_link.to_s()
                    next_pathunit.is_reverse = is_reverse
                    next_pathunits.append(next_pathunit)
                    pathunit_end_ss.append(most_likely_lane_link.from_s())
                    
                    
        if len(next_pathunits)!=0:
            if pathunit.is_junction(): # 无需完善end_s信息
                map_path.append(pathunit)
                cur_length = kJunctionLength
                for next_pathunit in next_pathunits:
                    self.dfs(next_pathunit, searched_length+cur_length, map_path, map_paths)
                map_path.pop()
            else: 
                for i in range(len(next_pathunits)):
                    pathunit.end_s = pathunit_end_ss[i]
                    next_pathunit = next_pathunits[i]
                    cur_length = pathunit.length()
                    map_path.append(pathunit)
                    self.dfs(next_pathunit, searched_length+cur_length, map_path, map_paths)
                    map_path.pop()
        else:
            if pathunit.is_junction():
                cur_length = kJunctionLength
                if searched_length + cur_length > kMinPathLength:
                    map_paths.append(map_path[:])
            else:
                cur_length = pathunit.length()
                if searched_length + cur_length > kMinPathLength:
                    map_path.append(pathunit)
                    map_paths.append(map_path[:])
                    map_path.pop()
        return
    
    
    def get_predecessor_lane(self, lane, s):
        max_s = float('-inf')
        most_likely_lane_link = None
        for lane_link in lane.predecessor_lane_links():
            from_lane = lane_link.from_lane()
            to_s = lane_link.to_s()
            if to_s > s: continue
            if from_lane.IsInJunction(): continue
            if to_s > max_s:
                max_s = to_s
                most_likely_lane_link = lane_link
        return most_likely_lane_link
    
    def get_successor_lane(self, lane, s):
        '''
        lane_link对象是在物理位置上处于lane和lane之间。存储了从哪个车道来from_lane和到哪个车道去to_lane
        还存储了从哪个车道来时离开from_lane的位置from_s（相对于旧车道而言）,和到哪个车道上去时的位置to_s（相对于新车道而言）
        '''
        min_s = float('inf')
        most_likely_lane_link = None
        for lane_link in lane.successor_lane_links():
            to_lane = lane_link.to_lane()
            from_s = lane_link.from_s()
            if from_s < s: continue
            if to_lane.IsInJunction(): continue
            if from_s < min_s:
                min_s = from_s
                most_likely_lane_link = lane_link
        return most_likely_lane_link
    
    # 当前位置在路口
    def get_driving_path_of_junction(self, junction):
        '''
        - desc:
            当前junction是start pathunit时进入该逻辑。返回next_pathunits
        - 逻辑：
            获取该junction出路口车道和进路口车道lane points(lane point对象包含lane和进入lane时的s)
            - 对于出口车道们(lane_point.s=0)
                - 过滤过窄的
                - 根据进入【出口车道】时的车道朝向lane_direction和自车ori的朝向差lane_diff_angle是否合理过滤其一
                - 过滤一个双向车道，保留另外一个
                - 根据lane_diff_angle和agent_point与车道中线的横向距离关系过滤
                - 根据离开junction的【出口车道】时的确定exit point与ori的agent_point的向量与agent_yaw向量的point_diff_angle角度过滤
                - 加入next_pathunits
            - 对于入口车道们(lane_point.s为lane_length)
                - 过滤过窄的
                - 过滤一个双向车道，保留另外一个
                - 将lane_heading+pi进行同上的判断
                - 加入reverse_pathunits
                - 根据angle的min关系，判断是否将reverse_pathunits加入next_pathunits
        '''
        next_pathunits = []
        lane_limit_angle = 5*math.pi/9 #100
        reverse_lane_limit_angle = math.pi/3 #60
        point_limit_angle = math.pi/2 #90

        min_point_diff_angle = 1e8
        out_lane_points = self.hdmap.GetDrivingOutOfJunctionLanes(junction.id())
        in_lane_points = self.hdmap.GetDrivingIntoJunctionLanes(junction.id())
        if len(out_lane_points) > 0 or len(in_lane_points) > 0:
            for lane_point in out_lane_points:
                left_width, right_width = lane_point.lane.GetWidth(lane_point.s)
                # 过滤过窄车道
                if left_width + right_width < 2.3:
                    continue
                lane_heading = lane_point.lane.GetHeading(lane_point.s)
                lane_diff_angle = self.normalize_angle(self.agent_yaw - lane_heading)
                
                # 过滤双向车道
                if lane_point.lane.bi_direction_lane() and abs(lane_diff_angle) > math.pi/2:
                    continue
                
                if abs(lane_diff_angle) > lane_limit_angle:
                    continue
                
                _, d = lane_point.lane.GetProjection(self.agent_point)
                # 过滤横向距离过远的直行车道
                if abs(lane_diff_angle) < math.pi/6:
                    if abs(d) > 15:
                        continue
                if lane_diff_angle > math.pi/18 and d > 1.0:
                    continue
                
                if lane_diff_angle < -math.pi/18 and d < -1.0:
                    continue
                    
                exit_point = Vec2d()
                lane_point.lane.GetPoint(lane_point.s, 0.0, exit_point)
                target_vec = exit_point - self.agent_point
                point_diff_angle = abs(self.normalize_angle(target_vec.Angle() - self.agent_yaw))
                
                if point_diff_angle < point_limit_angle:
                    next_pathunit = PathUnit(lane=lane_point.lane, start_s=lane_point.s, is_reverse=False)
                    next_pathunits.append(next_pathunit)
                    if point_diff_angle < min_point_diff_angle:
                        min_point_diff_angle = point_diff_angle
            
            reverse_pathunits = []
            is_reverse_angle_min = False
            is_reverse_d_min = False
            for lane_point in in_lane_points:
                left_width, right_width = lane_point.lane.GetWidth(lane_point.s)
                # 过滤过窄车道
                if left_width + right_width < 2.3:
                    continue
                lane_heading = lane_point.lane.GetHeading(lane_point.s)
                lane_diff_angle = self.normalize_angle(self.agent_yaw - lane_heading)
                # 过滤双向车道
                if lane_point.lane.bi_direction_lane() and abs(lane_diff_angle) > math.pi/2:
                    continue
                
                lane_heading += math.pi
                lane_diff_angle = self.normalize_angle(self.agent_yaw - lane_heading)
                
                if abs(lane_diff_angle) > reverse_lane_limit_angle:
                    continue
                    
                _, d = lane_point.lane.GetProjection(self.agent_front_point)
                # 过滤横向距离过远的直行车道
                if abs(lane_diff_angle) < math.pi/6 and abs(d)>15:
                    continue
                
                if lane_diff_angle > math.pi/18 and d<-1.0:
                    continue
                if lane_diff_angle < -math.pi/18 and d>1.0:
                    continue
                exit_point = Vec2d()
                lane_point.lane.GetPoint(lane_point.s, 0.0, exit_point)
                target_vec = exit_point - self.agent_point
                point_diff_angle = abs(self.normalize_angle(target_vec.Angle() - self.agent_yaw))
                if point_diff_angle < point_limit_angle:
                    next_pathunit = PathUnit(lane=lane_point.lane, start_s=lane_point.s, is_reverse=True)
                    reverse_pathunits.append(next_pathunit)
                    if point_diff_angle < min_point_diff_angle:
                        is_reverse_angle_min = True
                    if abs(d) < 10.0:
                        is_reverse_d_min = True
            if (is_reverse_angle_min and is_reverse_d_min) or len(next_pathunits)==0:
                next_pathunits.extend(reverse_pathunits)
        else:
            pass
            # print("Can not get the lanes outof(into) cur_junction: {}".format(junction.id().to_string()))
            
        return next_pathunits   
    
     
    # 后续进入路口
    def get_exit_path_of_junction(self, junction, pre_pathunit):
        '''
        - desc:
            当前junction存在前继pe_pathunit时进入该逻辑。返回next_pathunits
        - 逻辑
            获取该junction出路口车道和进路口车道lane points(lane point对象包含lane和进入lane时的s)
            - 对于出口车道
                - 过滤过窄
                - 过滤pre pathunit
                - 获取lane_diff_angle，不再是之前的agent yaw和lane_heading，而是pre_lane_heading和lane_heading
                - 双向车道
                - 角度限制
                - 横向距离（用injunction point）
                横向距离
        '''
        in_junction_point = Vec2d()# 从前继lane进入junction的点
        pre_pathunit.lane.GetPoint(pre_pathunit.end_s, 0.0, in_junction_point)
        pre_lane_heading = pre_pathunit.lane.GetHeading(pre_pathunit.end_s) # injunctuon的朝向
        is_reverse_in = pre_pathunit.is_reverse
        # 逆向进路口获取终点的反方向
        if is_reverse_in:
            pre_lane_heading += math.pi

        next_pathunits = []
        lane_limit_angle = 5*math.pi/9
        reverse_lane_limit_angle = math.pi/6
        
        out_lane_points = self.hdmap.GetDrivingOutOfJunctionLanes(junction.id())
        in_lane_points = self.hdmap.GetDrivingIntoJunctionLanes(junction.id())
        if len(out_lane_points) > 0 or len(in_lane_points) > 0:
            for lane_point in out_lane_points:
                left_width, right_width = lane_point.lane.GetWidth(lane_point.s)
                # 过滤过窄车道
                if left_width + right_width < 2.3:
                    continue
                    
                # 过滤同一个车道
                if is_reverse_in and lane_point.lane.id().value()==pre_pathunit.lane.id().value():
                    continue
                    
                lane_heading = lane_point.lane.GetHeading(lane_point.s)
                lane_diff_angle = abs(self.normalize_angle(lane_heading - pre_lane_heading))                
                # 过滤双向车道
                if lane_point.lane.bi_direction_lane() and lane_diff_angle > math.pi/2:
                    continue
                    
                if lane_diff_angle > lane_limit_angle:
                    continue
                
                # 过滤横向距离过远的直行车道
                if lane_diff_angle < math.pi/6:
                    _, d = lane_point.lane.GetProjection(in_junction_point)
                    if abs(d) > 15:
                        continue
       
                next_pathunit = PathUnit(lane=lane_point.lane, start_s=lane_point.s, is_reverse=False)
                next_pathunits.append(next_pathunit)

            if len(next_pathunits) == 0 or is_reverse_in:
                for lane_point in in_lane_points:
                    left_width, right_width = lane_point.lane.GetWidth(lane_point.s)
                    # 过滤过窄车道
                    if left_width + right_width < 2.3:
                        continue
                        
                    # 过滤同一个车道
                    if not is_reverse_in and lane_point.lane.id().value()==pre_pathunit.lane.id().value():
                        continue
                        
                    lane_heading = lane_point.lane.GetHeading(lane_point.s)
                    lane_diff_angle = abs(self.normalize_angle(lane_heading - pre_lane_heading))
                    # 过滤双向车道
                    if lane_point.lane.bi_direction_lane() and lane_diff_angle > math.pi/2:
                        continue
                    lane_heading += math.pi
                    lane_diff_angle = abs(self.normalize_angle(lane_heading - pre_lane_heading))
                    
                    if lane_diff_angle > reverse_lane_limit_angle:
                        continue
                        
                    # 过滤横向距离过远的直行车道
                    if lane_diff_angle < math.pi/6:
                        _, d = lane_point.lane.GetProjection(in_junction_point)
                        if abs(d) > 15:
                            continue
                    next_pathunit = PathUnit(lane=lane_point.lane, start_s=lane_point.s, is_reverse=True)
                    next_pathunits.append(next_pathunit)      
        else:
            pass
            # print("Can not get the lanes outof(into) next_junction: {}".format(junction.id().to_string()))
        return next_pathunits 
    
    def sample_candidate_points(self, map_paths):
        candidate_points = []
        self.sample_lane_candidate_points(map_paths, candidate_points)
        self.sample_junction_candidate_points(map_paths, candidate_points)
        return candidate_points
    
    def sample_lane_candidate_points(self, map_paths, candidate_points):
        '''
        对于每个map_path和其中的每个pathunit-lane，从中按照1m间隔对pathunit进行中线的采样（范围是lane的starts和ends之间）
        - 求采样点距离agent_point的距离，如果大于了最大搜索半径则标记finish，如果太近则continue，如果角度差太大finish
        '''
        point = Vec2d()
        unique_pathunits = set()
        for map_path in map_paths: # map_path
            finish_flag = False
            for pathunit in map_path:# pathunit
                if pathunit.is_junction():
                    continue
                if pathunit in unique_pathunits:
                    continue
                unique_pathunits.add(pathunit)
                s = pathunit.start_s
                if not pathunit.is_reverse:
                    while s < pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        dist = (point - self.agent_point).Length()
                        s += self.distance
                        if dist > self.search_range[1]:
                            finish_flag = True
                            break
                        elif dist < self.search_range[0]:
                            continue
                        if not self.is_valid_candidate_point(point): 
                            finish_flag = True
                            break
                        candidate_points.append([point.x(), point.y()])
                else:
                    while s > pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        dist = (point - self.agent_point).Length()
                        s -= self.distance
                        if dist > self.search_range[1]:
                            finish_flag = True
                            break
                        elif dist < self.search_range[0]:
                            continue
                        if not self.is_valid_candidate_point(point):
                            finish_flag = True
                            break
                        candidate_points.append([point.x(), point.y()]) 
                if finish_flag:
                    break
    
    def sample_junction_candidate_points(self, map_paths, candidate_points):
        '''
        对mappaths中的mappath中的junction进行采样
        '''
        junctions_dict = {}
        # 生成易于处理的格式
        for map_path in map_paths:# map_path
            for i in range(len(map_path)):
                pathunit = map_path[i] # pathunit
                if pathunit.is_junction():
                    if pathunit not in junctions_dict:
                        junctions_dict[pathunit] = {"in_junction_states":[], "longitudinal_pathunits":[], "lateral_pathunits":[]}
                    if i > 0: #该map_path的路径：lane-junction-lane 获取前继lane的信息
                        point = Vec2d()
                        in_pathunit = map_path[i-1]
                        in_pathunit.lane.GetPoint(in_pathunit.end_s, 0.0, point)
                        lane_heading = in_pathunit.lane.GetHeading(in_pathunit.end_s)
                        if in_pathunit.is_reverse:
                            lane_heading += math.pi
                        junctions_dict[pathunit]["in_junction_states"].append([point, lane_heading])
                        # if len(junctions_dict[pathunit]["in_junction_states"]) == 1:
                        #     in_pathunit.print_info()
                    if i+1 < len(map_path):# 该map_path的路径： junction-lane or lane-junction-lane 获取后继信息
                        out_pathunit = map_path[i+1]
                        in_heading = self.agent_yaw # 如果为junction-lane形式，则in_heading为agent_yaw
                        if len(junctions_dict[pathunit]["in_junction_states"]) > 0:#如果为lane-junction-lane格式，则inheading为前继lane的 lane heading
                            in_heading = junctions_dict[pathunit]["in_junction_states"][0][1] # 0 改为-1？
                        out_heading = out_pathunit.lane.GetHeading(out_pathunit.start_s)
                        if out_pathunit.is_reverse:
                            out_heading += math.pi
                        diff_angle = self.normalize_angle(out_heading - in_heading)
                        if abs(diff_angle) < math.pi/4:
                            junctions_dict[pathunit]["longitudinal_pathunits"].append(out_pathunit)
                        else:
                            junctions_dict[pathunit]["lateral_pathunits"].append(out_pathunit)
        # 采样
        for key, value in junctions_dict.items():
            in_point = self.agent_point if len(value["in_junction_states"]) == 0 else value["in_junction_states"][0][0]
            straight_points, is_sample_in_path = self.extend_longitudinal_pathunit_points(value["longitudinal_pathunits"], in_point, candidate_points)
            self.extend_lateral_pathunit_points(value["lateral_pathunits"], straight_points, in_point, is_sample_in_path, candidate_points)
            # print("in_point:{}, {}".format(in_point.x(), in_point.y()))
            # for pathunit in value["right_out_pathunits"]:
            #     pathunit.print_info()
                
    def extend_longitudinal_pathunit_points(self, pathunits, in_point, candidate_points):
        '''
            - desc: 当出junction的lane(pathunit)是直行或者小变道时的junction内点采样
            - input: 
                - pathunits: 出junction直行或者变道的pathunit
                - in_point: 进junction时前继lane的进入点 （没有前继lane则是agent的当前点agent point）

        '''
        straight_points = []
        unique_pathunits = set()
        extend_point = Vec2d()
        is_sample_in_path = True
        for pathunit in pathunits:
            if pathunit in unique_pathunits:
                continue
            unique_pathunits.add(pathunit)
            s,d = pathunit.lane.GetProjection(in_point)# lane-junction-lane模式下的前继lane的进入点
            if abs(d) < self.distance/2:
                is_sample_in_path = False
                 
            straight_point = Vec2d()
            pathunit.lane.GetPoint(pathunit.start_s, 0.0, straight_point)
            straight_points.append(straight_point)
            start_s = min(s, pathunit.start_s)
            end_s = max(s, pathunit.start_s) - self.distance
            while end_s > start_s:
                pathunit.lane.GetPoint(end_s, 0.0, extend_point)
                dist = (extend_point - self.agent_point).Length()
                end_s -= self.distance
                if dist > self.search_range[0] and dist < self.search_range[1]:
                    if not self.is_valid_candidate_point(extend_point):
                        continue
                    candidate_points.append([extend_point.x(), extend_point.y()])
        return straight_points, is_sample_in_path
           
        
    def extend_lateral_pathunit_points(self, pathunits, straight_points, in_point, is_sample_in_path, candidate_points):
        '''
            - desc: 当出junction的lane(pathunit)是转弯时的junction内点采样

        '''
        unique_pathunits = set()
        extend_point = Vec2d()
        for pathunit in pathunits:
            if pathunit in unique_pathunits:
                continue
            unique_pathunits.add(pathunit)
            s_ranges = self.get_multi_s_ranges(pathunit, straight_points, in_point, is_sample_in_path)#对于一个侧向pathunit
            for s_range in s_ranges:
                start_s, end_s = s_range[0], s_range[1]
                if pathunit.is_reverse:
                    while start_s <= end_s:
                        pathunit.lane.GetPoint(start_s, 0.0, extend_point)
                        dist = (extend_point - self.agent_point).Length()
                        start_s += self.distance
                        if dist > self.search_range[0] and dist < self.search_range[1]:
                            if not self.is_valid_candidate_point(extend_point):
                                continue
                            candidate_points.append([extend_point.x(), extend_point.y()])
                else:
                    while start_s >= end_s:
                        pathunit.lane.GetPoint(start_s, 0.0, extend_point)
                        dist = (extend_point - self.agent_point).Length()
                        start_s -= self.distance
                        if dist > self.search_range[0] and dist < self.search_range[1]:
                            if not self.is_valid_candidate_point(extend_point):
                                continue
                            candidate_points.append([extend_point.x(), extend_point.y()])
    
    def get_multi_s_ranges(self, pathunit, straight_points, in_point, is_sample_in_path):
        is_reverse = pathunit.is_reverse
        straight_ss = []
        s_ranges = []
        in_s,_ = pathunit.lane.GetProjection(in_point)
        for straight_point in straight_points:
            straight_s,_ = pathunit.lane.GetProjection(straight_point)
            straight_ss.append((straight_s, False))
        straight_ss.append((in_s, True)) # straight_ss存储了in_point/straight_points在侧向pathunit上的投影s（标志是否是in_point）
        if not is_reverse:
            straight_ss.sort(key=lambda x : x[0], reverse=True)# 降序
            pt_s = pathunit.start_s - self.distance# 侧向pathunit的采样开始点
            for s_info in straight_ss:
                if pt_s > s_info[0]:
                    if s_info[1]:
                        if is_sample_in_path:
                            s_ranges.append([pt_s, s_info[0]])
                        else:
                            s_ranges.append([pt_s, s_info[0]+self.distance])
                        pt_s = s_info[0] - self.distance
                    else:
                        s_ranges.append([pt_s, s_info[0]+self.distance])
                        pt_s = s_info[0] - self.distance
                else:
                    if s_info[1]:
                        if is_sample_in_path:
                            pt_s = s_info[0]
                        else:
                            pt_s = s_info[0] - self.distance
                    else:
                        pt_s = s_info[0] - self.distance
        else:
            straight_ss.sort(key=lambda x : x[0], reverse=False)
            pt_s = pathunit.start_s + self.distance
            for s_info in straight_ss:
                if pt_s < s_info[0]:
                    if s_info[1]:
                        if is_sample_in_path:
                            s_ranges.append([pt_s, s_info[0]])
                        else:
                            s_ranges.append([pt_s, s_info[0]-self.distance])
                        pt_s = s_info[0] + self.distance
                    else:
                        s_ranges.append([pt_s, s_info[0]-self.distance])
                        pt_s = s_info[0] + self.distance
                else:
                    if s_info[1]:
                        if is_sample_in_path:
                            pt_s = s_info[0]
                        else:
                            pt_s = s_info[0] + self.distance
                    else:
                        pt_s = s_info[0] + self.distance
            
        return s_ranges

#     def get_multi_s_ranges(self, pathunit, straight_points, in_point, flag=False):
#         s,_ = pathunit.lane.GetProjection(in_point)
#         straight_ss = []
#         s_ranges = []
            
#         for straight_point in straight_points:
#             straight_s,_ = pathunit.lane.GetProjection(straight_point)
#             straight_ss.append(straight_s)
#         if not pathunit.is_reverse:
#             straight_ss.sort(reverse= True)
#             start_s = pathunit.start_s - self.distance
#             for straight_s in straight_ss:
#                 if straight_s >= start_s:
#                     continue
#                 elif start_s <= s:
#                     break
#                 s_ranges.append([start_s, straight_s+self.distance])
#                 start_s = straight_s - self.distance
#             if start_s > s:
#                 s_ranges.append([start_s, s+self.distance]) 
#         else:
#             straight_ss.sort(reverse= False)
#             start_s = pathunit.start_s + self.distance
#             for straight_s in straight_ss:
#                 if straight_s <= start_s:
#                     continue
#                 elif start_s >= s:
#                     break
#                 s_ranges.append([start_s, straight_s-self.distance])
#                 start_s = straight_s + self.distance
#             if start_s < s:
#                 s_ranges.append([start_s, s-self.distance]) 
#         if flag:
#             print(s)
#             print(straight_ss)
#         return s_ranges

    def is_valid_candidate_point(self, point):
        target_vec = point - self.agent_point
        point_diff_angle = abs(self.normalize_angle(target_vec.Angle() - self.agent_yaw))
        return point_diff_angle < math.pi/2
        
                    

if __name__=='__main__':

    # ori = [404645.198, 3295265.55, 6.0, 0.90, 20, 5] # 已经进入路口
    # ori = [404635.198, 3295255.55, 6.0, 0.95, 20, 5] # 路口前
    ori = [404635.198, 3295255.55, 12.0, 0.95, 20, 5] # 路口前
    # ori = [404755.198, 3295375.55, 6.0, 0.90, 20, 5] # 普通直行直行
    hdmap = map_utils.get_hdmap()
    map_point_seacher = MapPointSeacher(hdmap)
    candidate_points = map_point_seacher.get_candidate_target_points(ori)
    candidate_refpaths, _, mappaths = map_point_seacher.get_candidate_refpaths(ori)
    
    plot_utils.draw_candidate_refpaths(ori, candidate_refpaths = candidate_refpaths)
    # plot_utils.draw_candidate_refpaths_multi(ori, candidate_refpaths = candidate_refpaths)
    plot_utils.draw_mappaths(ori, mappaths)

    print(f"总共有{len(candidate_points)}个候选点")
    plot_utils.draw_candidate_points(ori, candidate_points=candidate_points)