import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm


import sys
sys.path.append("/wg_dev/instance_centric_model/")
from src.data_loader import InterDataSet

dataset_dir = "/private/wanggang/instance_centric_data/"
set_name = "train"
dataset = InterDataSet(dataset_dir, set_name)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=8, collate_fn=dataset.collate_batch)
for input_dict in tqdm(dataloader):  
    pred_mask = (input_dict['candidate_mask'].sum(dim=-1) > 0).cuda() # B, N    
    gt_tar_offset = input_dict["gt_tar_offset"].cuda() # B, N ,2
    try:
        gt_tar_offset_1 = gt_tar_offset[pred_mask] # S, 2
    except Exception as e:
        print('产生错误了:',e)
        print(f"pred_mask.shape: {pred_mask.shape}")
        print(f"gt_tar_offset: {gt_tar_offset.shape}")
        print("pred_mask:")
        print(pred_mask)
        print("gt_tar_offset:")
        print(gt_tar_offset)

    
