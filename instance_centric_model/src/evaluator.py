import sys
sys.path.append('/data/wangchen/instance_centric/instance_centric_model')
sys.path.append('/data/wangchen/instance_centric/')
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import os
import datetime
import tqdm
from parser_args import  main_parser_for_evel, log_args_to_file
from collections import defaultdict
from utils import print_model_summary, find_trainable_layers, add_dict_prefix, formatted_time,create_logger, init_dist_pytorch, set_seed
from src.data_loader import get_ddp_dataloader

class Evaluator(object):
    def __init__(self, args, logger, test_latency=False):
        self.args = args
        self.logger = logger
        self.tb_log = SummaryWriter(log_dir=args.tensorboard_dir) if args.local_rank == 0 and self.args.enable_log else None
        
        self.data_loaders = dict()
        self.samplers = dict()
        for set_name in ['train', 'test']:
            self.data_loaders[set_name], self.samplers[set_name] = get_ddp_dataloader(args, set_name=set_name, test_latency=test_latency)
            
        self.net = self._initialize_network(args.model_name)
        if not args.without_sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(self.args.local_rank) 
        # 判断加载已有模型还是重新训练
        # if self.args.train_part == "back" or self.args.train_part == "joint":# load and freeze front part,   tran back part
        #     self._load_or_restart(self.args.train_part)
        #     self.start_epoch = 1
        # else: # front
        #     self.start_epoch = self._load_or_restart(self.args.train_part)
        self.net = self.net.cuda()
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, 
                                                    device_ids=[self.args.local_rank],find_unused_parameters=False)
        self.net.eval()


    
    def _initialize_network(self, model_name: str):
        """
        Import and initialize the requested model
        """
        if model_name == 'MODEL':
            from src.model import Model
            net = Model(self.args)
        else:
            raise NotImplementedError(
                f"Model architecture {model_name} does not exist yet.")

        # save net architecture to txt
        with open(os.path.join(self.args.model_dir, 'net.txt'), 'w') as f:
            f.write(str(net))

        return net
    


    def _load_and_freeze_front_checkpoint_train_back_part(self, load_checkpoint):
        """
        1. load_front_part_checkpoint
        2. freeze_front_part
        3. train back part
        """
        print("*"*100,"\n加载并冻结模型的前半部分，开始训练后半部分\n","*"*100)
        if load_checkpoint is not None:
            print("\nSaved model path:", load_checkpoint)
            # Load model（多进程需要放到cpu）
            if os.path.isfile(load_checkpoint):
                print('Loading checkpoint ...')
                checkpoint = torch.load(load_checkpoint,
                                         map_location=torch.device('cpu'))
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(
                    checkpoint['model_state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                for name, parameter in self.net.named_parameters():
                    parameter.requires_grad = False
                for module in self.net.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.eval()  # 确保不更新内部统计数据
                        module.weight.requires_grad = False
                        module.bias.requires_grad = False
                    elif isinstance(module, nn.LayerNorm):
                        module.eval()  # 确保不更新内部统计数据
                        module.weight.requires_grad = False
                        module.bias.requires_grad = False
                from .layers.score_decoder import ScoreDecoder
                self.net.scorer = ScoreDecoder(n_order=self.args.bezier_order)
                # return model_epoch
                return 0
            else:
                raise ValueError("No such pre-trained model:", load_checkpoint)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')

    def _load_checkpoint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        if load_checkpoint is not None:
            # Create load model path
            if load_checkpoint == 'best':
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_best_model.pt')
            else:  # Load specific checkpoint
                assert int(load_checkpoint) > 0, \
                    "Check args.load_model. Must be an integer > 0"
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_epoch_' +
                    str(load_checkpoint).zfill(3) + '.pt')
            print("\nSaved model path:", saved_model_name)
            # Load model（多进程需要放到cpu）
            if os.path.isfile(saved_model_name):
                print('Loading checkpoint ...')
                checkpoint = torch.load(saved_model_name,
                                         map_location=torch.device('cpu'))
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(
                    checkpoint['model_state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                # return model_epoch
                return 0
            else:
                raise ValueError("No such pre-trained model:", saved_model_name)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')
        
    def _load_checkpoint_joint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        print("*"*100,"\n联合训练全部结构和loss\n","*"*100)
        if load_checkpoint is not None:
            print("\nSaved model path:", load_checkpoint)
            # Load model（多进程需要放到cpu）
            if os.path.isfile(load_checkpoint):
                print('Loading checkpoint ...')
                checkpoint = torch.load(load_checkpoint,
                                         map_location=torch.device('cpu'))
                model_epoch = checkpoint['epoch']
                def add_module_prefix(state_dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict['module.' + k] = v
                    return new_state_dict
                
                self.net.load_state_dict(
                    add_module_prefix(checkpoint['model_state_dict']))
                print('Loaded checkpoint at epoch', model_epoch, '\n')
            else:
                raise ValueError("No such pre-trained model:", load_checkpoint)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')
        return model_epoch
    
    def _load_checkpoint_eval(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        print("*"*100,"\n eval模式，加载pt文件进模型 \n","*"*100)
        if load_checkpoint is not None:
            print("\nSaved model path:", load_checkpoint)
            if os.path.isfile(load_checkpoint):
                print('Loading checkpoint ...')
                checkpoint = torch.load(load_checkpoint,
                                         map_location=torch.device('cpu'))
                model_epoch = checkpoint['epoch']
                def add_module_prefix(state_dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict['module.' + k] = v
                    return new_state_dict
                
                self.net.load_state_dict(
                    add_module_prefix(checkpoint['model_state_dict']))
                print('Loaded checkpoint at epoch', model_epoch, '\n')
            else:
                raise ValueError("No such pre-trained model:", load_checkpoint)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')
        return model_epoch

    def _load_or_restart(self, train_part, pt):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if pt is not None:
            if train_part == "back":
                loaded_epoch = self._load_and_freeze_front_checkpoint_train_back_part(pt)
            elif train_part == "front":
                loaded_epoch = self._load_checkpoint(pt)
            elif train_part == "joint":# joint
                loaded_epoch = self._load_checkpoint_joint(pt)
            else:
                loaded_epoch = self._load_checkpoint_eval(pt)
            # start from the following epoch
            start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
        return start_epoch
    

    def train(self):
        start_epoch = self.start_epoch
        # if self.args.train_part == "back":
        #     start_epoch = 1
        # elif self.args.train_part == "front":
        #     start_epoch = self._load_or_restart("front")
        # else:
        #     start_epoch = self._load_or_restart("joint")
        


        if self.args.local_rank==0:
            # 输出模型的参数信息
            print_model_summary(self.net, self.args.model_name)
            # 要更新的参数
        params = find_trainable_layers(self.net)
        # 设置优化器
        # 设置学习率变化器
        
        # 开始训练
        self._train_loop(start_epoch=start_epoch, end_epoch=self.args.num_epochs)

    def _evaluate_loop(self, pts):
        phase_name = 'test'
        torch.cuda.empty_cache()
        self.net.eval()
        for pt in pts:
            torch.cuda.empty_cache()
            self._evaluate_pt(pt)

    # 整个epoch的训练
    def _train_loop(self, start_epoch, end_epoch):        
        phase_name = 'train'
         # 初始化学习率
        if self.scheduler is not None:
            learning_rate = self.optimizer.param_groups[0]['lr']
        else:
            learning_rate = self.args.learning_rate
        for cur_epoch in range(start_epoch, end_epoch+1):
            start_time = time.time()  # time epoch
            torch.cuda.empty_cache()
            self.samplers['train'].set_epoch(cur_epoch)
            # train one epoch
            train_losses = self._train_epoch(cur_epoch)

            if self.args.local_rank == 0:
                loss_str = ', '.join([f"{loss_name}={loss_value:.5f}" for loss_name, loss_value in train_losses.items()])
                self.logger.info(f'----Epoch {cur_epoch}, time/epoch={formatted_time(time.time() - start_time)}, learning_rate={learning_rate:.5f}, {loss_str}')
                if self.tb_log is not None:
                    self.tb_log.add_scalar('meta_data/learning_rate', learning_rate, cur_epoch)
                    for key, val in train_losses.items():
                        self.tb_log.add_scalar('train/' + key, val, cur_epoch)
                if cur_epoch % self.args.save_every == 0:
                    self._save_checkpoint(cur_epoch)  # save model
                    self.logger.info(f"Saved latest checkpoint at epoch {cur_epoch}")

             # 更新学习速率，暂时不采用
            if self.scheduler is not None:
                if self.args.scheduler == 'ReduceLROnPlateau':
                    pass
                # TODO(wg）
                    # if epoch >= self.args.start_validation and \
                    #         epoch % self.args.validate_every == 0:
                    #     lr_sched_metric = test_metrics["test_TOP_1"]
                    #     self.scheduler.step(lr_sched_metric)
                elif self.args.scheduler in ['ExponentialLR',
                                             'CosineAnnealingLR']:
                    self.scheduler.step()
                learning_rate = self.optimizer.param_groups[0]['lr']
                
            # 验证
            if cur_epoch >= self.args.start_validation and cur_epoch % self.args.validate_every == 0:
            # if True:
                torch.cuda.empty_cache()
                self._evaluate_pt(cur_epoch, mode='test')
 
    # 单个epoch的训练
    def _train_epoch(self, epoch):
        self.net.train()
        # losses_epoch = {"loss":0, "ref_cls_loss": 0, "traj_loss": 0, "score_loss": 0, "plan_reg_loss":0, "plan_score_loss":0,"irl_loss":0,"weights_regularization":0}
        losses_epoch = {"loss":0, "ref_cls_loss": 0, "traj_loss": 0, "score_loss": 0, "plan_reg_loss":0, "irl_loss":0,"weights_regularization":0}
        total_it_each_epoch = len(self.data_loaders['train'])
        dataloader_iter = iter(self.data_loaders['train'])
        with tqdm.trange(0, total_it_each_epoch, desc='train_epoch', dynamic_ncols=True, leave=(self.args.local_rank == 0)) as pbar:
            for cur_it in pbar:
                try:
                    input_dict = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self.data_loaders['train'])
                    input_dict = next(dataloader_iter)
                    print('new iters')
                self.optimizer.zero_grad()  # sets grads to zero
                output_dict = self.net(input_dict)
                 # 计算模型损失
                loss, loss_dict = self.loss(input_dict, output_dict, epoch)

                losses_epoch["loss"] += loss.detach().item()
                losses_epoch["ref_cls_loss"] += loss_dict["ref_cls_loss"].detach().item()
                losses_epoch["traj_loss"] += loss_dict["traj_loss"].detach().item()
                losses_epoch["score_loss"] += loss_dict["score_loss"].detach().item()
                # if "safety_loss" in loss_dict:
                #     losses_epoch["safety_loss"] += loss_dict["safety_loss"].detach().item()
                losses_epoch["plan_reg_loss"] += loss_dict["plan_reg_loss"].detach().item()
                # losses_epoch["plan_score_loss"] += loss_dict["plan_score_loss"].detach().item()
                losses_epoch["irl_loss"] += loss_dict["irl_loss"].detach().item()
                losses_epoch["weights_regularization"] += loss_dict["weights_regularization"].detach().item()

                loss.backward()
                # 暂时不适用梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
                self.optimizer.step()
        if self.args.local_rank == 0:
            for loss_name in losses_epoch.keys():
                losses_epoch[loss_name] = losses_epoch[loss_name]/total_it_each_epoch
            losses_epoch = add_dict_prefix(losses_epoch, prefix="train")
        return losses_epoch
    
    @torch.no_grad()
    def _evaluate_pt(self, pt, mode='test'):
        """
        在测试集上对模型进行验证，计算准确率和对应的损失
        """
        print(f"evaluate {pt}")
        self._load_or_restart("eval", pt)
        self.net.eval()  # evaluation mode
        res_dict = defaultdict(lambda: torch.tensor(0.0).cuda())

        total_iter_each_epoch = len(self.data_loaders[mode])
        dataloader_iter = iter(self.data_loaders[mode])
        with torch.no_grad():
            with tqdm.trange(0, total_iter_each_epoch, desc='eval_epochs', dynamic_ncols=True, leave=(self.args.local_rank == 0)) as pbar:
                for cur_it in pbar:# 一个循环是一个iter，所有的iter组成一个epoch，会有dataset_num/batch_size = iterion次循环前向传播.这里计算是将每个iter的metric累加，最后求平均
                    try:
                        input_dict = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(self.data_loaders['test'])
                        input_dict = next(dataloader_iter)
                    output_dict = self.net(input_dict)
                    
                    metric_dict = self.net.module.compute_model_metrics(input_dict, output_dict)
                    for key,val in metric_dict.items():
                        res_dict[key] += torch.as_tensor(val[0],device='cuda')


        dist.barrier()

        for key in res_dict:
            dist.reduce(res_dict[key], 0)
  
        if self.args.local_rank == 0:
            for key in res_dict:
                res_dict[key] = res_dict[key].item()
            for key in res_dict:
                if metric_dict[key][1] == 1:
                    res_dict[key] = res_dict[key]/ res_dict['valid_batch_size']
                elif metric_dict[key][1] == 2:
                    res_dict[key] = res_dict[key]/ res_dict['batch_size']

            text = f"Eval: {pt}, "
            for key,val in res_dict.items():
                text+=f"{key}={val:.5f},  "

            self.logger.info(text)

            # if traj_ade < self.best_ade:
            #     self.best_ade = traj_ade
            #     self._save_checkpoint(epoch, best_epoch=True)  # save model
            #     self.logger.info(f"Saved best checkpoint at epoch {epoch}")

    @torch.no_grad()
    def test_latency(self, pt, mode='test'):
        """
        在测试集上对模型进行验证，计算准确率和对应的损失
        """
        print(f"evaluate {pt} 's latency")
        torch.cuda.empty_cache()
        self._load_or_restart("eval", pt)
        self.net.eval()  # evaluation mode
        res_dict = defaultdict(lambda: torch.tensor(0.0).cuda())


        with torch.no_grad():

            torch.manual_seed(42)
            ego_refpath_num = 2
            agent_refpath_num = 5
            target_agent_num = 2
            all_agent_num = 1
            map_elem_num = 1
            instance_num = all_agent_num + map_elem_num
            circle_num = 600
            times = torch.zeros(circle_num) # 一次iter 的耗时
            device = torch.device("cuda")
            input_dict = {"agent_ctrs":torch.rand(1,all_agent_num,2, device=device),
                          "agent_vecs":torch.rand(1,all_agent_num,2, device=device),
                          "agent_feats":torch.rand(1,all_agent_num,20,13, device=device),
                          "agent_mask":torch.randint(1,2,(1,all_agent_num,20), device=device),

                          "ego_refpath_cords":torch.rand(1, ego_refpath_num, 20, 2, device=device),
                          "ego_refpath_vecs":torch.rand(1, ego_refpath_num, 20, 2, device=device),
                          "ego_vel_mode":torch.randint(1,3,(1,), device=device),
                          "ego_gt_cand":torch.rand(1,ego_refpath_num, device=device),
                          "ego_gt_traj":torch.rand(1,50,2, device=device),
                          "ego_cand_mask": torch.randint(1,2,(1, ego_refpath_num), device=device),

                          "candidate_refpaths_cords":torch.rand(1, all_agent_num, agent_refpath_num, 20, 2, device=device),
                          "candidate_refpaths_vecs":torch.rand(1, all_agent_num, agent_refpath_num, 20, 2, device=device),
                          "gt_preds":torch.rand(1,all_agent_num,50,2, device=device),
                          "gt_vel_mode":torch.randint(1,3,(1, all_agent_num), device=device),
                          "gt_candts":torch.rand(1,all_agent_num, agent_refpath_num, device=device),
                          "candidate_mask":torch.randint(1,2,(1, all_agent_num, agent_refpath_num), device=device),

                          "map_ctrs":torch.rand(1, map_elem_num, 2, device=device),
                          "map_vecs":torch.rand(1, map_elem_num, 2, device=device),
                          "map_feats":torch.rand(1, map_elem_num, 20, 5, device=device),
                          "map_mask":torch.randint(1,2,(1, map_elem_num, 20), device=device),
                          "rpe":torch.rand(1, instance_num, instance_num, 5, device=device),
                          "rpe_mask":torch.randint(1,2,(1, instance_num, instance_num), device=device),
                          }
            for cur_it in tqdm.tqdm(range(circle_num)):# 一个循环是一个iter，所有的iter组成一个epoch，会有dataset_num/batch_size = iterion次循环前向传播.这里计算是将每个iter的metric累加，最后求平均
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                output_dict = self.net(input_dict)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times[cur_it] = end_time - start_time
                # scenes[cur_it] = input_dict['agent_ctrs'].shape[0]
                # target_actors[cur_it] = torch.sum(input_dict['candidate_mask'].sum(-1) > 0) # b,N,M -> b,N -> 1
                # actors[cur_it] = torch.sum(input_dict['agent_mask'].sum(-1) > 0) # b,N,20 -> b,N -> 1
                # mapors[cur_it] = torch.sum(input_dict['map_mask'].sum(-1) > 0) # b,N,20 -> b,N -> 1
                # instances[cur_it] = mapors[cur_it] + actors[cur_it]
                
                # metric_dict = self.net.module.compute_model_metrics(input_dict, output_dict)
                # for key,val in metric_dict.items():
                #     res_dict[key] += torch.as_tensor(val[0],device='cuda')
            print(times)
            print(times[10:].mean(-1).item())

    @torch.no_grad()
    def test_latency_for_instances(self, pt, mode='test'):
        logger.info(f"\ninstance_num,all_agent_num,map_elem_num,agent_refpath_num,mean_time")

        """
        在测试集上对模型进行验证，计算准确率和对应的损失
        """
        print(f"evaluate {pt} 's latency")
        torch.cuda.empty_cache()
        self._load_or_restart("eval", pt)
        self.net.eval()  # evaluation mode
        res_dict = defaultdict(lambda: torch.tensor(0.0).cuda())


        torch.manual_seed(42)
        with torch.no_grad():
            for instance_num in range(20,200):
                all_agent_num = int(instance_num*0.22)
                map_elem_num = instance_num - all_agent_num
                agent_refpath_num = int(instance_num/12)
                ego_refpath_num = 2
                # agent_refpath_num = 5
                target_agent_num = 2
                # all_agent_num = 1
                # map_elem_num = 1
                # instance_num = all_agent_num + map_elem_num
                circle_num = 600
                times = torch.zeros(circle_num) # 一次iter 的耗时
                device = torch.device("cuda")
                input_dict = {"agent_ctrs":torch.rand(1,all_agent_num,2, device=device),
                            "agent_vecs":torch.rand(1,all_agent_num,2, device=device),
                            "agent_feats":torch.rand(1,all_agent_num,20,13, device=device),
                            "agent_mask":torch.randint(1,2,(1,all_agent_num,20), device=device),

                            "ego_refpath_cords":torch.rand(1, ego_refpath_num, 20, 2, device=device),
                            "ego_refpath_vecs":torch.rand(1, ego_refpath_num, 20, 2, device=device),
                            "ego_vel_mode":torch.randint(1,3,(1,), device=device),
                            "ego_gt_cand":torch.rand(1,ego_refpath_num, device=device),
                            "ego_gt_traj":torch.rand(1,50,2, device=device),
                            "ego_cand_mask": torch.randint(1,2,(1, ego_refpath_num), device=device),

                            "candidate_refpaths_cords":torch.rand(1, all_agent_num, agent_refpath_num, 20, 2, device=device),
                            "candidate_refpaths_vecs":torch.rand(1, all_agent_num, agent_refpath_num, 20, 2, device=device),
                            "gt_preds":torch.rand(1,all_agent_num,50,2, device=device),
                            "gt_vel_mode":torch.randint(1,3,(1, all_agent_num), device=device),
                            "gt_candts":torch.rand(1,all_agent_num, agent_refpath_num, device=device),
                            "candidate_mask":torch.randint(1,2,(1, all_agent_num, agent_refpath_num), device=device),

                            "map_ctrs":torch.rand(1, map_elem_num, 2, device=device),
                            "map_vecs":torch.rand(1, map_elem_num, 2, device=device),
                            "map_feats":torch.rand(1, map_elem_num, 20, 5, device=device),
                            "map_mask":torch.randint(1,2,(1, map_elem_num, 20), device=device),
                            "rpe":torch.rand(1, instance_num, instance_num, 5, device=device),
                            "rpe_mask":torch.randint(1,2,(1, instance_num, instance_num), device=device),
                            }
                for cur_it in tqdm.tqdm(range(circle_num)):# 一个循环是一个iter，所有的iter组成一个epoch，会有dataset_num/batch_size = iterion次循环前向传播.这里计算是将每个iter的metric累加，最后求平均
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    output_dict = self.net(input_dict)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    times[cur_it] = end_time - start_time
                    # scenes[cur_it] = input_dict['agent_ctrs'].shape[0]
                    # target_actors[cur_it] = torch.sum(input_dict['candidate_mask'].sum(-1) > 0) # b,N,M -> b,N -> 1
                    # actors[cur_it] = torch.sum(input_dict['agent_mask'].sum(-1) > 0) # b,N,20 -> b,N -> 1
                    # mapors[cur_it] = torch.sum(input_dict['map_mask'].sum(-1) > 0) # b,N,20 -> b,N -> 1
                    # instances[cur_it] = mapors[cur_it] + actors[cur_it]
                    
                    # metric_dict = self.net.module.compute_model_metrics(input_dict, output_dict)
                    # for key,val in metric_dict.items():
                    #     res_dict[key] += torch.as_tensor(val[0],device='cuda')
                logger.info(f"{instance_num},{all_agent_num},{map_elem_num},{agent_refpath_num},{times[10:].mean(-1).item()}")
                # print(times)
                # print(times[10:].mean(-1).item())


 
        
if __name__ == "__main__":
    args = main_parser_for_evel()
    total_gpus, args.local_rank = init_dist_pytorch(args.local_rank, backend='nccl')
    if args.reproducibility:
        set_seed(seed_value=7777)
    # pt_dir = "/private/wangchen/instance_model/output/MODEL/2024-07-02 23:45:29_front(最新15w数据训练得到)/saved_models"
    # pt_dir = "/private/wangchen/instance_model/output/MODEL/2024-06-28 17:13:02_back/saved_models"
    pt_dir = "/private/wangchen/instance_model/my/output/MODEL/2024-07-12 08:54:33_front_6/saved_models"
    pts = [os.path.join(pt_dir, pt_file) for pt_file in os.listdir(pt_dir)]
    pts.sort()
    pts = pts[::-1]
    pts = [pt for pt in pts if pt.count('030') > 0]
    # pts = [pt for pt in pts if pt.count('010') > 0 or pt.count('030') > 0 or pt.count('050') > 0]
    log_file = args.log_dir + "/log_train_%s.txt"%datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file, args.local_rank)
    log_args_to_file(args, logger)
    


    piple = Evaluator(args,logger, test_latency=False)
    piple._evaluate_loop(pts=pts)
    # piple.test_latency_for_instances(pt=pts[0]) # !!!!!!!!! attenion gpu set to 1
            

    