import matplotlib.pyplot as plt
import numpy as np
from modules.hdmap_lib.python.binding.libhdmap import HDMapManager
import common.plot_config as plot_config
import math, copy
from pathlib import Path
import common.time_utils as time_utils
from modules.hdmap_lib.python.binding.libhdmap import Vec2d
import common.map_utils as map_utils







def draw_all_lanes(axes, lanes, color='r'):
    DELTA_S = 2
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
        axes.plot(xs, ys, color=color, ls=':')


def draw_all_lanes_(ax):
    # map_bin_path = plot_config.map_bin_path
    # business_scene = 'port_meishan'

    # HDMapManager.LoadMap(map_bin_path, business_scene)
    # hdmap = HDMapManager.GetHDMap()
    hdmap = map_utils.get_hdmap()
    
    draw_all_lanes(ax,hdmap.GetMap().lanes(), color='lightcoral')



def draw_points(points, draw_map = True):
    fig, axes = plt.subplots(figsize=(9,9), dpi= 800)
    average_x = sum(point.x() for point in points) / len(points)
    average_y = sum(point.y() for point in points) / len(points)
    axes.set_xlim(average_x - 120,average_x + 120)
    axes.set_ylim(average_y - 120, average_y + 120)
    if draw_map:
        map_bin_path = plot_config.map_bin_path
        business_scene = 'port_meishan'

        HDMapManager.LoadMap(map_bin_path, business_scene)
        hdmap = HDMapManager.GetHDMap()
        draw_all_lanes(axes,hdmap.GetMap().lanes(), color='lightcoral')
        
    for point in points:
        axes.plot(point.x(), point.y(), marker ='x', ms = 15, color= 'r')
    
    fig.savefig('tmp.jpg')
    print("draw complete")








def get_xy_lim(ori, candidate_points = None, candidate_refpaths = None, points_xy = None, extend = 15):
    if candidate_points != None:
        candidate_points = np.asarray(candidate_points)
        min_x = min(ori[0], np.min(candidate_points[:, 0]))
        max_x = max(ori[0], np.max(candidate_points[:, 0]))
        min_y = min(ori[1], np.min(candidate_points[:, 1]))
        max_y = max(ori[1], np.max(candidate_points[:, 1]))
    elif candidate_refpaths != None:
        min_x, max_x = ori[0], ori[0]
        min_y, max_y = ori[1], ori[1]
        for refpath in candidate_refpaths:
            refpath = np.asarray(refpath)
            min_x = min(min_x, np.min(refpath[:, 0]))
            max_x = max(max_x, np.max(refpath[:, 0]))
            min_y = min(min_y, np.min(refpath[:, 1]))
            max_y = max(max_y, np.max(refpath[:, 1]))
    elif points_xy != None:
        points_x, points_y = copy.deepcopy(points_xy)
        points_x = list(points_x)
        points_y = list(points_y)
        points_x.append(ori[0])
        points_y.append(ori[1])
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)
    else:
        print("error")
        exit()

    return [min_x-extend, max_x+extend, min_y-extend, max_y+extend]


        
def calculate_box_corners(ori):
    half_length = 2
    half_width = 1
    corner_xs, corner_ys = [], []
    # right head
    corner_xs.append(ori[0] + half_length * math.cos(ori[3]) + half_width * math.cos(ori[3] - 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[3]) + half_width * math.sin(ori[3] - 0.5*math.pi))
    # left head
    corner_xs.append(ori[0] + half_length * math.cos(ori[3]) + half_width * math.cos(ori[3] + 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[3]) + half_width * math.sin(ori[3] + 0.5*math.pi))
    # left tail
    corner_xs.append(ori[0] + half_length * math.cos(ori[3] + math.pi) + half_width * math.cos(ori[3] + 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[3] + math.pi) + half_width * math.sin(ori[3] + 0.5*math.pi))
    # right tail
    corner_xs.append(ori[0] + half_length * math.cos(ori[3] + math.pi) + half_width * math.cos(ori[3] - 0.5*math.pi))
    corner_ys.append(ori[1] + half_length * math.sin(ori[3] + math.pi) + half_width * math.sin(ori[3] - 0.5*math.pi))
    
    corner_xs.append(corner_xs[0])
    corner_ys.append(corner_ys[0])
    return corner_xs, corner_ys
        

# 绘制候选点
def draw_candidate_points(ori, candidate_points, output_dir="tmp"):
    fig, ax = plt.subplots(figsize=(22,12))
    ax.axis("equal")
    roi_matrix = get_xy_lim(ori, candidate_points=candidate_points)
    ax.axis(roi_matrix)
    # draw_all_lanes(ax, hdmap.GetMap().lanes(), color='lightcoral')
    # 地图
    draw_all_lanes_(ax)
    # 车位置
    ax.scatter([ori[0]], [ori[1]], c='y', marker='o', alpha=0.8)
    # 车头框
    corner_xs, corner_ys = calculate_box_corners(ori)
    ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)
    # 候选点
    candidate_points = np.asarray(candidate_points)
    ax.scatter(candidate_points[:,0], candidate_points[:,1], c='b', marker='o', alpha=0.8)

    output_dir= Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"mk dir {output_dir}")
    fig_save_path = output_dir / f"candiadate_points{time_utils.get_cur_time_string()}.jpg"
    fig.savefig(fig_save_path)
    print(f"draw candiadate_points at {fig_save_path.resolve()}")
    # plt.show()
     

# def draw_one_refpath_one_pic(idx, refpath, color,roi_matrix):
#     fig, ax = plt.subplots(figsize=(22,12),dpi=800)
#     ax.axis("equal")
#     draw_all_lanes_(ax)
#     ax.scatter([ori[0]], [ori[1]], c='y', marker='o', alpha=0.8)
    
#     corner_xs, corner_ys = calculate_box_corners(ori)
#     ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)
#     ax.text(refpath[0][0],refpath[0][1], "s", color=color)
#     ax.plot([xy[0] for xy in refpath], [xy[1] for xy in refpath], color=color, alpha=0.5, linewidth=1)
#     ax.text(refpath[-1][0],refpath[-1][1], "e", color=color)
#     ax.axis(roi_matrix)
#     fig.savefig(f"{idx}'s refpath")


def draw_candidate_refpaths_with_his_fut(ori, candidate_refpaths, his_traj, fut_traj, output_dir="tmp"):
    fig, ax = plt.subplots(figsize=(22,12),dpi=800)
    draw_candidate_refpaths(ori, candidate_refpaths,my_ax=ax)
    ax.axis('equal')
    traj_all = np.concatenate((his_traj, fut_traj), axis=0)
    roi_matrix = get_xy_lim(ori, points_xy=(traj_all[:,0], traj_all[:,1]))
    ax.axis(roi_matrix)
    ax.plot(his_traj[:,0], his_traj[:,1], color='red',zorder=15,alpha= 0.5)
    ax.plot(fut_traj[:,0], fut_traj[:,1], marker='x', linestyle="--", color="green",alpha=0.5)
    ax.scatter(fut_traj[49,0], fut_traj[49,1], color='purple', zorder=15)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"mkdir {output_dir}")
    fig_save_path = output_dir / f'candidate_refpath_with_his_fut{time_utils.get_cur_time_string()}.jpg'
    fig.savefig(fig_save_path)
    print(f"draw candiadate_refpath at {fig_save_path.resolve()}")

def draw_candidate_refpaths(ori, candidate_refpaths, output_dir="tmp", my_ax = None):
    '''
        candidate_refpaths: list[refpath]
        refpath: [(x1,y2),(x2,y2)]
    '''
    if my_ax == None: 
        fig, ax = plt.subplots(figsize=(22,12),dpi=800)
    else:
        ax = my_ax

    ax.axis("equal")
    roi_matrix = get_xy_lim(ori, candidate_refpaths=candidate_refpaths)
    ax.axis(roi_matrix)
    ax.set_title(f"total {len(candidate_refpaths)} candidate_refpaths")


    # 地图
    draw_all_lanes_(ax)
    # 车点
    ax.scatter([ori[0]], [ori[1]], c='y', marker='o', alpha=0.8)
    # 车头
    corner_xs, corner_ys = calculate_box_corners(ori)
    ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)

    colors = plt.cm.viridis(np.linspace(0, 1, len(candidate_refpaths)))

    for idx, (refpath, color) in enumerate(zip(candidate_refpaths,colors)):
        # draw_one_refpath_one_pic(idx, refpath, color,roi_matrix)

        ax.text(refpath[0][0],refpath[0][1], f"{idx}'s", color=color)
        ax.plot([xy[0] for xy in refpath], [xy[1] for xy in refpath], color=color, alpha=0.5, linewidth=1)
        ax.text(refpath[-1][0],refpath[-1][1], f"{idx}'e", color=color)

    if my_ax == None: 
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            print(f"mkdir {output_dir}")
        fig_save_path = output_dir / f'candidate_refpath{time_utils.get_cur_time_string()}.jpg'
        fig.savefig(fig_save_path)
        print(f"draw candiadate_refpath at {fig_save_path.resolve()}")

    # plt.show()

def draw_candidate_refpaths_multi(ori, candidate_refpaths, output_dir="tmp"):
    '''
        candidate_refpaths: list[refpath]
        refpath: [(x1,y2),(x2,y2)]
    '''
    num_candidate_refpaths = len(candidate_refpaths)
    total_fig_num = math.ceil(num_candidate_refpaths/9)
    roi_matrix = get_xy_lim(ori, candidate_refpaths=candidate_refpaths)

    colors = plt.cm.viridis(np.linspace(0, 1, len(candidate_refpaths)))
    # 车头
    corner_xs, corner_ys = calculate_box_corners(ori)
    for fig_i in range(total_fig_num):
        fig, axes = plt.subplots(3,3,figsize=(22,12),dpi=1200)
        
        for i in range(3):
            for j in range(3):
                sub_fig_num = fig_i * 9 + i * 3 + j # 第sub_fig_num个ref path
                if sub_fig_num >= num_candidate_refpaths:
                    break
                ax = axes[i, j]
                ax.axis("equal")
                ax.axis(roi_matrix)
                ax.set_title(f"the {sub_fig_num} candidate_refpath")
                # 地图
                draw_all_lanes_(ax)
                # 车点
                ax.scatter([ori[0]], [ori[1]], c='y', marker='o', alpha=0.8)
                ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)
                refpath = candidate_refpaths[sub_fig_num]
                color = colors[sub_fig_num]

                ax.text(refpath[0][0],refpath[0][1], f"s", color=color)
                ax.plot([xy[0] for xy in refpath], [xy[1] for xy in refpath], color=color, alpha=0.5, linewidth=1)
                ax.text(refpath[-1][0],refpath[-1][1], f"e", color=color)
            if sub_fig_num >= num_candidate_refpaths:
                    break
            
        
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            print(f"mkdir {output_dir}")
        fig.suptitle(f'candidate_refpath num of {fig_i} fig [{fig_i*9}, {sub_fig_num}]\ntotal {num_candidate_refpaths}')
        
        fig_save_path = output_dir / f'candidate_refpath num of {fig_i} fig [{fig_i*9}, {sub_fig_num}]_{time_utils.get_cur_time_string()}.jpg'
        fig.savefig(fig_save_path)
        print(f"multi fig for draw candidate ref path save at {fig_save_path}")


    print(f"draw candiadate_refpath at {output_dir.resolve()}")

    # plt.show()


def draw_mappaths(ori, mappaths, output_dir = "tmp"):
    point = Vec2d()
    point_x = []
    point_y = []
    junction_point_x = []
    junction_point_y = []
    for mappath in mappaths:
        for pathunit in mappath:
            if pathunit.is_junction():
                junction_points = pathunit.junction.polygon().points()
                junction_point_x.extend([point.x() for point in junction_points])
                junction_point_y.extend([point.y() for point in junction_points])
                
            else:
                s = pathunit.start_s
                if pathunit.is_reverse:
                    while s > pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        point_x.append(point.x())
                        point_y.append(point.y())
                        s -= 0.1
                else:
                    while s < pathunit.end_s:
                        pathunit.lane.GetPoint(s, 0.0, point)
                        point_x.append(point.x())
                        point_y.append(point.y())
                        s += 0.1

    fig,ax = plt.subplots(figsize=(10,10),dpi=800)
    draw_all_lanes_(ax)
    ax.axis('equal')
    roi_matrix = get_xy_lim(ori,points_xy=(point_x, point_y))
    ax.axis(roi_matrix)
    # pathunit-lanes
    ax.plot(point_x, point_y, "o r", ms = 1,zorder = 10)
    # pathunit-junctions
    ax.plot(junction_point_x, junction_point_y, ".-b", zorder= 5)
    # 车点
    ax.scatter(ori[0],ori[1], marker='X',color="purple")
    # 车头
    corner_xs, corner_ys = calculate_box_corners(ori)
    ax.plot(corner_xs, corner_ys, color='b', linewidth=2.0)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    fig_path = output_dir/ f"map_paths_{time_utils.get_cur_time_string()}.jpg"
    fig.savefig(fig_path)
    print(f"mappath draw at {fig_path.resolve()}")
                
    pass # TODO:wangchen 可视化一下mappath，寻找路口前很近的距离没法生成轨迹的原因

