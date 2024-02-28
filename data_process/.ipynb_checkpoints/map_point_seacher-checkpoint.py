import math
import numpy as np
from modules.hdmap_lib.python.binding.libhdmap import Vec2d
import matplotlib.pyplot as plt

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
            candidate_points = self.sample_candidate_points(map_paths)
            # print(f"总共有{len(map_paths)}条map_path")
            # for map_path in map_paths:
            #     print("##"*12)
            #     for pathunit in map_path:
            #         pathunit.print_info()

            if len(candidate_points) == 0:
                candidate_points = self.get_candidate_target_points_base_grid(ori)
        if len(candidate_points) == 0:
            candidate_points = self.get_candidate_target_points_base_grid(ori)
            return candidate_points
        return candidate_points
    
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
            if not lane.IsInJunction() and not lane.bi_direction_lane():
                pathunit = PathUnit(lane=lane)
                pathunits = self.set_reverse_and_s(pathunit)
                start_pathunits.extend(pathunits)
                break
            elif lane.bi_direction_lane():
                lane_s, _ = lane.GetProjection(self.agent_point)
                lane_heading = lane.GetHeading(lane_s)
                diff_angle = self.normalize_angle(search_yaw - lane_heading)
                if abs(diff_angle) > math.pi/2:
                    continue
                else:
                    pathunit = PathUnit(lane=lane)
                    pathunits = self.set_reverse_and_s(pathunit)
                    start_pathunits.extend(pathunits)
                    break
        is_filter_reverse = False
        for pathunit in start_pathunits:
            if not pathunit.is_junction() and not pathunit.is_reverse:
                is_filter_reverse = True
                break
        lanes = self.hdmap.GetLanes(self.agent_front_point, self.width)
        junctions = self.hdmap.GetJunctions(self.agent_front_point, self.width)
        for lane in lanes:
            if lane.id().value() in unique_pathunit_ids:
                continue
            unique_pathunit_ids.add(lane.id().value())
            if not lane.IsInJunction() and not lane.bi_direction_lane():
                pathunit = PathUnit(lane=lane)
                pathunits = self.set_reverse_and_s(pathunit)
                if is_filter_reverse and len(pathunits) == 1 and pathunits[0].is_reverse:
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
    
    def set_reverse_and_s(self, pathunit):
        if pathunit.is_junction():
            return [pathunit]
        is_reverse = False
        # 障碍物xy坐标转化成sd坐标
        s,_ = pathunit.lane.GetProjection(self.agent_point)
        lane_heading = pathunit.lane.GetHeading(s)
        pathunit.start_s = s
        diff_angle = abs(self.normalize_angle(lane_heading - self.agent_yaw))
        # 双向都要搜索
        if diff_angle > math.pi/3 and diff_angle < 2*math.pi/3:
            reverse_pathunit = PathUnit()
            reverse_pathunit.lane = pathunit.lane
            reverse_pathunit.start_s = pathunit.start_s
            reverse_pathunit.is_reverse = True
            pathunit.is_reverse = False
            return [pathunit, reverse_pathunit]
        elif diff_angle <= math.pi/3:
            pathunit.is_reverse = False
            return [pathunit]
        else:
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

            # 没有获取到前后继,获取后续路口
            if most_likely_lane_link == None:
                lane_end_point = Vec2d()
                lane_s = 0.0 if is_reverse else pathunit.lane.length()
                pathunit.lane.GetPoint(lane_s, 0.0, lane_end_point)
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
            else:
                if is_reverse:
                    next_pathunit = PathUnit()
                    lane = most_likely_lane_link.from_lane()
                    next_pathunit.lane = lane
                    next_pathunit.start_s = most_likely_lane_link.from_s()
                    next_pathunit.is_reverse = is_reverse
                    pathunit_end_ss.append(most_likely_lane_link.to_s())
                    next_pathunits.append(next_pathunit)
                else:
                    next_pathunit = PathUnit()
                    lane = most_likely_lane_link.to_lane()
                    next_pathunit.lane = lane
                    next_pathunit.start_s = most_likely_lane_link.to_s()
                    next_pathunit.is_reverse = is_reverse
                    next_pathunits.append(next_pathunit)
                    pathunit_end_ss.append(most_likely_lane_link.from_s())
                    
                    
        if len(next_pathunits)!=0:
            if pathunit.is_junction():
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
        next_pathunits = []
        lane_limit_angle = 5*math.pi/9
        reverse_lane_limit_angle = math.pi/3
        point_limit_angle = math.pi/2

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
        in_junction_point = Vec2d()
        pre_pathunit.lane.GetPoint(pre_pathunit.end_s, 0.0, in_junction_point)
        pre_lane_heading = pre_pathunit.lane.GetHeading(pre_pathunit.end_s)
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
        point = Vec2d()
        unique_pathunits = set()
        for map_path in map_paths:
            finish_flag = False
            for pathunit in map_path:
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
        junctions_dict = {}
        # 生成易于处理的格式
        for map_path in map_paths:
            for i in range(len(map_path)):
                pathunit = map_path[i]
                if pathunit.is_junction():
                    if pathunit not in junctions_dict:
                        junctions_dict[pathunit] = {"in_junction_states":[], "longitudinal_pathunits":[], "lateral_pathunits":[]}
                    if i > 0:
                        point = Vec2d()
                        in_pathunit = map_path[i-1]
                        in_pathunit.lane.GetPoint(in_pathunit.end_s, 0.0, point)
                        lane_heading = in_pathunit.lane.GetHeading(in_pathunit.end_s)
                        if in_pathunit.is_reverse:
                            lane_heading += math.pi
                        junctions_dict[pathunit]["in_junction_states"].append([point, lane_heading])
                        # if len(junctions_dict[pathunit]["in_junction_states"]) == 1:
                        #     in_pathunit.print_info()
                    if i+1 < len(map_path):
                        out_pathunit = map_path[i+1]
                        in_heading = self.agent_yaw
                        if len(junctions_dict[pathunit]["in_junction_states"]) > 0:
                            in_heading = junctions_dict[pathunit]["in_junction_states"][0][1]
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
        straight_points = []
        unique_pathunits = set()
        extend_point = Vec2d()
        is_sample_in_path = True
        for pathunit in pathunits:
            if pathunit in unique_pathunits:
                continue
            unique_pathunits.add(pathunit)
            s,d = pathunit.lane.GetProjection(in_point)
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
        unique_pathunits = set()
        extend_point = Vec2d()
        for pathunit in pathunits:
            if pathunit in unique_pathunits:
                continue
            unique_pathunits.add(pathunit)
            s_ranges = self.get_multi_s_ranges(pathunit, straight_points, in_point, is_sample_in_path)
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
        straight_ss.append((in_s, True))
        if not is_reverse:
            straight_ss.sort(key=lambda x : x[0], reverse=True)
            pt_s = pathunit.start_s - self.distance
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
        
                    

def draw_all_lanes(ax, lanes, color='r'):
    DELTA_S = 4
    lanes_map = dict()
    if isinstance(lanes, list):
        for lane in lanes:
            lanes_map[lane.id().value()] = lane
    elif isinstance(lanes, dict):
        lanes_map = lanes
    else:
        assert False, "lanes input param is {} which is not dict or list".format(
            type(lanes))
    for lane_id, lane in lanes_map.items():
        ref_line = lane.reference_line()
        xs, ys = list(), list()
        for s in np.arange(0.0, ref_line.length(), DELTA_S):
            ref_point = ref_line.GetReferencePoint(s)
            xs.append(ref_point.x())
            ys.append(ref_point.y())
        ax.plot(xs, ys, color=color, ls=':')
        
def calculate_box_corners(ori):
    half_length = 2
    half_width = 1
    corner_xs, corner_ys = [], []
    # right head
    corner_xs.append(ori[0] + half_length * math.cos(ori[-1]) + half_width * math.cos(ori[-1] - 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[-1]) + half_width * math.sin(ori[-1] - 0.5*math.pi))
    # left head
    corner_xs.append(ori[0] + half_length * math.cos(ori[-1]) + half_width * math.cos(ori[-1] + 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[-1]) + half_width * math.sin(ori[-1] + 0.5*math.pi))
    # left tail
    corner_xs.append(ori[0] + half_length * math.cos(ori[-1] + math.pi) + half_width * math.cos(ori[-1] + 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[-1] + math.pi) + half_width * math.sin(ori[-1] + 0.5*math.pi))
    # right tail
    corner_xs.append(ori[0] + half_length * math.cos(ori[-1] + math.pi) + half_width * math.cos(ori[-1] - 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[-1] + math.pi) + half_width * math.sin(ori[-1] - 0.5*math.pi))
    
    corner_xs.append(corner_xs[0])
    corner_ys.append(corner_ys[0])
    return corner_xs, corner_ys
        
def draw_candidate_points(hdmap, ori, candidate_points):
    fig, ax = plt.subplots(figsize=(22,12))
    ax.axis("equal")
    draw_all_lanes(ax, hdmap.GetMap().lanes(), color='lightcoral')
    ax.scatter([ori[0]], [ori[1]], c='y', marker='o', alpha=0.8)
    
    corner_xs, corner_ys = calculate_box_corners(ori)
    ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)
    min_width, min_height = ori[0], ori[1]
    max_width, max_height = ori[0], ori[1]
    candidate_points = np.asarray(candidate_points)
    ax.scatter(candidate_points[:,0], candidate_points[:,1], c='b', marker='o', alpha=0.8)
    min_width = min(min_width, np.min(candidate_points[:, 0]))
    max_width = max(max_width, np.max(candidate_points[:, 0]))
    min_height = min(min_height, np.min(candidate_points[:, 1]))
    max_height = max(max_height, np.max(candidate_points[:, 1]))
    roi_matrix = [min_width - 5, max_width + 5, min_height - 5, max_height + 5]
    ax.axis(roi_matrix)
    
    plt.show()
     

if __name__=='__main__':
    from modules.hdmap_lib.python.binding.libhdmap import HDMapManager
    map_file_path = "/wg_dev/fabu_projects/hdmap/map_data/20230411/meishangang/map.bin"
    scene_type = 'port_meishan'
    HDMapManager.LoadMap(map_file_path, scene_type)
    hdmap = HDMapManager.GetHDMap()
    ori = [404640.198, 3295319.55, 6.0, 2]
    map_point_seacher = MapPointSeacher(hdmap)
    candidate_points = map_point_seacher.get_candidate_target_points(ori)
    print(f"总共有{len(candidate_points)}个候选点")
    draw_candidate_points(hdmap, ori, candidate_points)