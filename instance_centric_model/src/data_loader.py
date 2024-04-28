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
            self.ids = self.ids[:2000000]
        else:
            self.ids = self.ids[:10000]
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        data_path = os.path.join(self.data_root, self.ids[i])
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
        return data_dict
    
    def collate_batch(self, batch_list):
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
        
        input_dict = {}
        for key, val_list in key_to_list.items():
            val_list = [torch.from_numpy(x) for x in val_list]
            if key in ['agent_feats', 'agent_mask', 'agent_ctrs', 'agent_vecs', 'gt_preds', 'gt_tar_offset',
                       'plan_feat', 'plan_mask', 'map_ctrs', 'map_vecs', 'map_feats', 'map_mask']:
                input_dict[key] = self.merge_batch_1d(val_list)
            elif key in ['tar_candidate', 'candidate_mask', 'gt_candts', 'rpe', 'rpe_mask']:
                input_dict[key] = self.merge_batch_2d(val_list)
            else:
                print(key)
                continue
        return input_dict
                
    def merge_batch_1d(self, tensor_list):
        assert len(tensor_list[0].shape) in [2, 3]
        only_2d_tensor = False
        if len(tensor_list[0].shape) == 2:
            tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
            only_2d_tensor = True
        tensor_list = [x.unsqueeze(dim=0) for x in tensor_list]
        max_feat0 = max([x.shape[1] for x in tensor_list])
        _, _, num_feat1, num_feat2 = tensor_list[0].shape
        ret_tensor_list = []
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, num_feat1, num_feat2)
            new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
            ret_tensor_list.append(new_tensor)

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
        if only_2d_tensor:
            ret_tensor = ret_tensor.squeeze(dim=-1)
        return ret_tensor
    
    def merge_batch_2d(self, tensor_list):
        assert len(tensor_list[0].shape) in [2, 3]
        only_2d_tensor = False
        if len(tensor_list[0].shape) == 2:
            tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
            only_2d_tensor = True
        tensor_list = [x.unsqueeze(dim=0) for x in tensor_list]
        max_feat0 = max([x.shape[1] for x in tensor_list])
        max_feat1 = max([x.shape[2] for x in tensor_list])
        
        num_feat2 = tensor_list[0].shape[-1]
        ret_tensor_list = []
        for k in range(len(tensor_list)):
            cur_tensor = tensor_list[k]
            new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, max_feat1, num_feat2)
            new_tensor[:, :cur_tensor.shape[1], :cur_tensor.shape[2], :] = cur_tensor
            ret_tensor_list.append(new_tensor)

        ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1_maxt, num_feat2)
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
                            pin_memory=True, num_workers=args.workers,
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




