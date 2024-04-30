import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import multiprocessing

def test_dataset(i):
    data_path = os.path.join(data_root, ids[i])
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    gt_tar_offset = torch.from_numpy(data_dict['gt_tar_offset'])
    has_infinite = ~torch.isfinite(gt_tar_offset).all()
    if has_infinite:
        print(data_path)
    return


if __name__=="__main__":
    dataset_dir = "/private/wanggang/instance_centric_data/"
    set_name = "train"
    data_root = os.path.join(dataset_dir, set_name)
    ids = os.listdir(data_root) 
    pool = multiprocessing.Pool(processes=16)
    pool.map(test_dataset, range(len(ids)))
    print("############完成###########")
    pool.close()
    pool.join()
