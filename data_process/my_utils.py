
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from datetime import datetime


def get_cur_time_string():
    return timestamp2string_more(time.time())

def timestamp2string(timestmap):
    struct_time = time.localtime(timestmap)
    str = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)
    return str

def timestamp2string_more(timestamp):
    # 将浮点型时间戳转换为 datetime 对象
    dt = datetime.fromtimestamp(timestamp)
    
    # 格式化 datetime 对象为字符串，包括毫秒
    # %f 代表微秒，取前3位得到毫秒
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def draw_traj(traj, text=None, output_dir="tmp"):
    assert isinstance(traj, np.ndarray), "变量不是ndarray类型"
    fig,ax = plt.subplots(figsize=(10,10),dpi=200)
    ax.axis('equal')

    ax.plot(traj[:,0],traj[:,1])
    ax.text(traj[-1,0],traj[-1,1], "e", fontsize=12)
    ax.text(traj[0,0],traj[0,1], "s", fontsize=12)

    if text != None:
        mean_xy = traj.mean(0)
        ax.text(mean_xy[0],mean_xy[1], text)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"mk dir {output_dir}")
    fig_save_path = output_dir / f"draw_traj{get_cur_time_string()}.jpg"
    fig.savefig(fig_save_path)
    print(f"draw candiadate_points at {fig_save_path.resolve()}")



def draw_all(data, output_dir="tmp"):
    fig,ax = plt.subplots(figsize=(10,10),dpi=200)
    ax.axis('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    orig = data['orig'] # obs坐标点
    theta = data['theta']

    agt_traj_obs_normal = data['agt_traj_obs_normal']
    agt_traj_fut_normal = data['agt_traj_fut_normal']

    feats = data['feats'] # # [object_num,20,3] x,y,mask      agent坐标系
    has_obss = data['has_obss'] # [object_num, 20]  标志20帧中的有效帧 （靠后的位置填满，不同于simpl）

    has_preds = data['has_preds'] #[object_num,30] 标志30帧中的有效帧
    gt_preds = data['gt_preds'] # [object_num,30,2] 该object后续30帧的真实轨迹     agent坐标系

    tar_candidate = data['tar_candts'] # [cand_n, 2] agent坐标系
    gt_candts = data['gt_candts'] # [cand,1] agent坐标系
    gt_tar_offset = data['gt_tar_offset'] # [2，] 采样点加偏移得到真实轨迹点GT

    graph = data['graph']
    ctrs = graph['ctrs'] # m,2
    lane_idcs = graph['lane_idcs'] # m 
    map_feats = graph['feats'] # m,2
    lane_idx = np.unique(lane_idcs)

    obj_num = feats.shape[0]
    obs_horizon = 20
    pred_horizon = 50


    # draw lane
    colors = plt.cm.viridis(np.linspace(0, 1, len(lane_idx)))
    for c, idx in zip(colors, lane_idx):
        index = np.where(lane_idcs == idx)[0]
        start_idx, end_idx = index[0], index[-1]

        lane_ctr = ctrs[start_idx:end_idx+1,:]
        
        lane_vec = map_feats[start_idx:end_idx+1,:]
        ax.plot(lane_ctr[:,0], lane_ctr[:,1],marker='.',color = c, ms = 4)
        # ax.quiver(lane_ctr[:,0], lane_ctr[:,1], lane_vec[:,0], lane_vec[:,1],scale=20, color = c)
    
    # agt_traj_obs_normal = data['agt_traj_obs_normal']
    # agt_traj_fut_normal = data['agt_traj_fut_normal']
    # ax.plot(agt_traj_obs_normal[:,0], agt_traj_obs_normal[:,1],color='cyan')
    # ax.plot(agt_traj_fut_normal[:,0], agt_traj_fut_normal[:,1],color='black')


    # draw obj
    for i in range(obj_num):
        feat = feats[i,:,:2]
        has_obs = has_obss[i]
        valid_feat = feat[has_obs] # v,2
        gt_pred = gt_preds[i]
        has_pred = has_preds[i]
        valid_pred = gt_pred[has_pred]
        if i == 0:
            ax.plot(valid_feat[:,0], valid_feat[:,1], color='tan') # obs
            ax.plot(valid_pred[:,0], valid_pred[:,1], color='blue') # fut
            ax.scatter(valid_feat[-1,0], valid_feat[-1,1], color='purple') # orig
            ax.text(valid_feat[-1,0], valid_feat[-1,1], "target agent") 
        else:
            ax.plot(valid_feat[:,0], valid_feat[:,1], color='tan') # obs
            ax.plot(valid_pred[:,0], valid_pred[:,1], color='blue') # fut
            ax.scatter(valid_feat[-1,0], valid_feat[-1,1], color='purple') # orig
            ax.text(valid_feat[-1,0], valid_feat[-1,1], "other agent") 

    # draw cand point
    ax.scatter(tar_candidate[:,0], tar_candidate[:,1], s= 8, color='red')
    gt_point = tar_candidate[gt_candts.reshape(-1).astype(bool)] # 1,2
    ax.scatter(gt_point[0,0],gt_point[0,1],color='green') # [cand_n, 2] -> [1,2]
    ax.text(gt_point[0,0],gt_point[0,1],"gt cand") # [cand_n, 2] -> [1,2]

    ax.scatter(gt_point[0,0] +gt_tar_offset[0] ,gt_point[0,1]+gt_tar_offset[1], color="yellow")
    ax.text(gt_point[0,0] +gt_tar_offset[0] ,gt_point[0,1]+gt_tar_offset[1],"of")



    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"mk dir {output_dir}")
    fig_save_path = output_dir / f"draw_all{get_cur_time_string()}.jpg"
    fig.savefig(fig_save_path)
    print(f"draw all at {fig_save_path.resolve()}")







            