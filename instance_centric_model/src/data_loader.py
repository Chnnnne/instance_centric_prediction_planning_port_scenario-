import os
import torch
import random
from torch.utils.data import Dataset,DataLoader
import pickle
from utils import get_dist_info

class InterDataSet(Dataset):
    def __init__(self, dataset_dir, set_name):
        self.data_root = os.path.join(dataset_dir, set_name)
        self.ids = os.listdir(self.data_root)
        random.seed(777)
        random.shuffle(self.ids)
        if set_name=="train":
            self.ids = self.ids[:50000]
        else:
            self.ids = self.ids[:20000]
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        data_path = os.path.join(self.data_root, self.ids[i])
        with open(data_path, "rb") as f:
            try:
                data_dict = pickle.load(f)
            except:
                print("something wroth occurs during load the pkl data, plz check\n"*30)
                exit()
                bak_up_file_path = "/private/wangchen/instance_model/instance_model_data/test/howo55_1691777795.6718624.pkl"
                with open(bak_up_file_path, "rb") as b:
                    data_dict = pickle.load(b)
        return data_dict
    
    def collate_batch(self, batch_list):
        '''
        将一个list的samples合成
        key_to_list['candidate_refpaths_cords'] = [sample-1 pad_candidate_refpaths_cords (all_n, max-N, 20,2)  , sample=2 pad_candidate_refpaths_cords, ...]
        '''
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)] #  key:key     val:list[val]
        
        input_dict = {}
        for key, val_list in key_to_list.items():
            val_list = [torch.from_numpy(x) for x in val_list]
            if key in ['agent_feats', 'agent_mask', 'agent_ctrs', 'agent_vecs', 'gt_preds',
                       'plan_feat', 'plan_mask', 'map_ctrs', 'map_vecs', 'map_feats', 'map_mask', 'gt_vel_mode']:
                input_dict[key] = self.merge_batch_1d(val_list)
            elif key in ['candidate_mask', 'gt_candts', 'rpe', 'rpe_mask']:
                input_dict[key] = self.merge_batch_2d(val_list)
            elif key in ['candidate_refpaths_cords', 'candidate_refpaths_vecs']:
                input_dict[key] = self.merge_batch_2d_more(val_list)
            elif key in ['ego_vel_mode', 'ego_refpath_cords', 'ego_refpath_vecs',"ego_gt_traj"]:
                input_dict[key] = torch.stack(val_list,dim=0) #(B,)
            else:
                print(key)
                continue
            '''
            val_list = [torch.from_numpy(x) for x in val_list]
            if key in ['agent_feats', 'agent_mask', 'agent_ctrs', 'agent_vecs', 'gt_preds', 'gt_tar_offset',
                       'plan_feat', 'plan_mask', 'map_ctrs', 'map_vecs', 'map_feats', 'map_mask']:
                input_dict[key] = self.merge_batch_1d(val_list)
            elif key in ['tar_candidate', 'candidate_mask', 'gt_candts', 'rpe', 'rpe_mask']:
                input_dict[key] = self.merge_batch_2d(val_list)
            else:
                print(key)
                continue
            '''
        return input_dict
                
    def merge_batch_1d(self, tensor_list):# agent feats:list[all_n,20,13]      agent mask: list[all_n,20]
        assert len(tensor_list[0].shape) in [1,2, 3]
        only_2d_tensor = False
        if len(tensor_list[0].shape) == 2:
            tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list] # [alln,20] -> [alln,20,1]      [bs,alln,20,1]
            only_2d_tensor = True

        only_1d_tensor = False
        if len(tensor_list[0].shape) == 1: # gt_vel_mode [all_n]
            tensor_list = [x.unsqueeze(dim=-1).unsqueeze(dim=-1) for x in tensor_list] # [all_n, 1, 1]
            only_1d_tensor = True

        tensor_list = [x.unsqueeze(dim=0) for x in tensor_list] #list[1, all_n,20,13]
        max_feat0 = max([x.shape[1] for x in tensor_list]) # all_n-Max
        _, _, num_feat1, num_feat2 = tensor_list[0].shape # [20,13]
        ret_tensor_list = []
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k] # [1, all_n,20,13]
            assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, num_feat1, num_feat2) # [1, all_n-Max,20,13]
            new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor # [1, all_n-Max,20,13]
            ret_tensor_list.append(new_tensor) # list[1, all_n-Max,20,13]

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # [bs, all_n-Max,20,13]
        if only_2d_tensor:
            ret_tensor = ret_tensor.squeeze(dim=-1)
        if only_1d_tensor:
            ret_tensor = ret_tensor.squeeze(dim=-1).squeeze(dim=-1)
        return ret_tensor
    
    def merge_batch_2d_more(self, tensor_list):
        assert len(tensor_list[0].shape) == 4
        tensor_list = [x.unsqueeze(dim=0) for x in tensor_list] # list[1, all_n, Max-N, 20, 2]
        max_feat0 = max([x.shape[1] for x in tensor_list]) # all_n-Max
        max_feat1 = max([x.shape[2] for x in tensor_list]) # Max-N-Max
        num_feat2 = tensor_list[0].shape[3] # 20
        num_feat3 = tensor_list[0].shape[4] # 2
        ret_tensor_list = []

        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, max_feat1, num_feat2, num_feat3) # 1, all_n-Max, Max-N-Max, 2
            new_tensor[:, :cur_tensor.shape[1], :cur_tensor.shape[2], :, :] = cur_tensor
            ret_tensor_list.append(new_tensor)

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # bs,all_n-Max, Max-N-Max, 20, 2
        return ret_tensor


    def merge_batch_2d(self, tensor_list): # tar_candidate:  list[all_n, Max-N, 2]
        assert len(tensor_list[0].shape) in [2, 3]
        only_2d_tensor = False
        if len(tensor_list[0].shape) == 2:
            tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
            only_2d_tensor = True
        tensor_list = [x.unsqueeze(dim=0) for x in tensor_list] # list[1, all_n, Max-N, 2]
        max_feat0 = max([x.shape[1] for x in tensor_list]) # all_n-Max
        max_feat1 = max([x.shape[2] for x in tensor_list]) # Max-N-Max
        
        num_feat2 = tensor_list[0].shape[-1] # 2
        ret_tensor_list = []
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, max_feat1, num_feat2) # 1, all_n-Max, Max-N-Max, 2
            new_tensor[:, :cur_tensor.shape[1], :cur_tensor.shape[2], :] = cur_tensor
            ret_tensor_list.append(new_tensor)

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # bs,all_n-Max, Max-N-Max, 2
        if only_2d_tensor:
            ret_tensor = ret_tensor.squeeze(dim=-1)
        return ret_tensor
                
def get_ddp_dataloader(args, set_name):
    assert set_name in ['train', 'test']
    def worker_init_fn_(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
    dataset = InterDataSet(args.dataset_dir, set_name)
    if set_name=='train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        drop_last = True 
    else:
        rank, world_size = get_dist_info()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
        drop_last = False
    if args.local_rank == 0:
        print(f"{set_name} dataset contains {len(dataset)} data")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            pin_memory=False, num_workers=args.workers,
                            shuffle=False, collate_fn=dataset.collate_batch,
                            drop_last=drop_last, sampler = sampler, timeout=0
                            )
    return dataloader, sampler

def get_dataloader_for_eval(args, set_name):
    assert set_name in ['train', 'test']
    dataset = InterDataSet(args.dataset_dir,set_name)
    batch_size = args.batch_size
    drop_last = True if set_name=='train' else False
    num_workers = args.workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, collate_fn=dataset.collate_batch)
    return dataloader




