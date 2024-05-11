import sys
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import time
import os
import tqdm

from utils import print_model_summary, find_trainable_layers, add_dict_prefix, formatted_time
from src.data_loader import get_ddp_dataloader
from src.loss import Loss

class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.tb_log = SummaryWriter(log_dir=args.tensorboard_dir) if args.local_rank == 0 and self.args.enable_log else None
        
        self.data_loaders = dict()
        self.samplers = dict()
        for set_name in ['train', 'test']:
            self.data_loaders[set_name], self.samplers[set_name] = get_ddp_dataloader(args, set_name=set_name)
            
        self.net = self._initialize_network(args.model_name)
        if not args.without_sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        torch.cuda.set_device(self.args.local_rank) 
        self.net = self.net.cuda()
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, 
                                                    device_ids=[self.args.local_rank])
        # find_unused_parameters=True
        
        self.optimizer = None
        self.scheduler = None
        self.loss = Loss()
        self.best_ade = float('inf') 
    
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
    
    def _set_optimizer(self, optimizer_name: str, parameters):
        """
        Set selected optimizer
        """
        if optimizer_name == 'AdamW':
            return  torch.optim.AdamW(
                parameters, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif optimizer_name == 'Adam':
            return torch.optim.Adam(
                parameters, lr=self.args.learning_rate)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(
                parameters, lr=self.args.learning_rate)
        else:
            raise NameError(f'Optimizer {optimizer_name} not implemented.')
            
    def _set_scheduler(self, scheduler_name: str):
        """
        Set selected scheduler
        """
        # ReduceOnPlateau
        if scheduler_name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                min_lr=1e-6,
                verbose=True)
        # Exponential
        elif scheduler_name == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99)
        # CosineAnnealing
        elif scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=30)
        else:  # Not set
            return None


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

    def _load_or_restart(self):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if self.args.load_checkpoint is not None:
            loaded_epoch = self._load_checkpoint(self.args.load_checkpoint)
            # start from the following epoch
            start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
        return start_epoch
    
    def _save_checkpoint(self, epoch, best_epoch=False):
        """
        保存模型和优化器状态
        """
        saved_models_path = os.path.join(self.args.model_dir, 'saved_models')
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        # Save current checkpoint
        if not best_epoch:
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_epoch_' +
                str(epoch).zfill(3) + '.pt')
        else:  # best model name
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_best_model.pt')
        # TODO(wg) save时加上module，使用于分布式保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.module.state_dict(),}, saved_model_name)
    
    def train(self):
        # 判断加载已有模型还是重新训练
        start_epoch = self._load_or_restart()
        if self.args.local_rank==0:
            # 输出模型的参数信息
            print_model_summary(self.net, self.args.model_name)
            # 要更新的参数
        params = find_trainable_layers(self.net)
        # 设置优化器
        self.optimizer = self._set_optimizer(self.args.optimizer, params)
        # 设置学习率变化器
        self.scheduler = self._set_scheduler(self.args.scheduler)
        self.logger.info(self.args.local_rank % torch.cuda.device_count())
        
        # 开始训练
        self._train_loop(start_epoch=start_epoch, end_epoch=self.args.num_epochs)
    
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
                self._evaluate_epoch(cur_epoch, mode='test')
 
    # 单个epoch的训练
    def _train_epoch(self, epoch):
        self.net.train()
        # losses_epoch = {"loss":0, "tar_cls_loss": 0, "tar_offset_loss": 0, "traj_loss": 0, "score_loss": 0, "safety_loss": 0}
        losses_epoch = {"loss":0, "ref_cls_loss": 0, "traj_loss": 0, "score_loss": 0, "safety_loss": 0}
        # total_it_each_epoch = int(0.1*len(self.data_loaders['train']))
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
                # losses_epoch["tar_offset_loss"] += loss_dict["tar_offset_loss"].detach().item()
                losses_epoch["traj_loss"] += loss_dict["traj_loss"].detach().item()
                losses_epoch["score_loss"] += loss_dict["score_loss"].detach().item()
                if "safety_loss" in loss_dict:
                    losses_epoch["safety_loss"] += loss_dict["safety_loss"].detach().item()
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
    def _evaluate_epoch(self, epoch, mode='test'):
        """
        在测试集上对模型进行验证，计算准确率和对应的损失
        """
        self.net.eval()  # evaluation mode
        total, top_acc, target_fde, traj_ade, traj_fde = torch.zeros(5).cuda()
        total_it_each_epoch = len(self.data_loaders[mode])
        dataloader_iter = iter(self.data_loaders[mode])
        with tqdm.trange(0, total_it_each_epoch, desc='eval_epochs', dynamic_ncols=True, leave=(self.args.local_rank == 0)) as pbar:
            for cur_it in pbar:
                try:
                    input_dict = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self.data_loaders['test'])
                    input_dict = next(dataloader_iter)
                output_dict = self.net(input_dict)
                batch_size, top, t_ade, t_fde = self.net.module.compute_model_metrics(input_dict, output_dict)
                total += torch.tensor(batch_size).cuda()
                top_acc += torch.tensor(top).cuda()
                # target_fde += torch.as_tensor(fde, device='cuda')
                traj_ade += torch.as_tensor(t_ade, device='cuda')
                traj_fde += torch.as_tensor(t_fde, device='cuda')
        dist.barrier()
        for x in [total, top_acc, traj_ade, traj_fde]:
            dist.reduce(x, 0) # reduce并返回给进程0
  
        if self.args.local_rank == 0:
            top_acc = 100 * top_acc.item() / total.item()
            # target_fde = target_fde.item() / total.item()
            traj_ade = traj_ade.item() / total.item()
            traj_fde = traj_fde.item() / total.item()
            self.logger.info(f'Eval: case_num={total.item()}, top_acc={top_acc:.2f}%, traj_ade={traj_ade:.5f}, traj_fde={traj_fde:.5f}')
            if traj_ade < self.best_ade:
                self.best_ade = traj_ade
                self._save_checkpoint(epoch, best_epoch=True)  # save model
                self.logger.info(f"Saved best checkpoint at epoch {epoch}")

    # @torch.no_grad()
    # def _evaluate_epoch(self, epoch, mode='test'):
    #     """
    #     在测试集上对模型进行验证，计算准确率和对应的损失
    #     """
    #     self.net.eval()  # evaluation mode
    #     total, top_acc, target_fde, traj_ade, traj_fde = torch.zeros(5).cuda()
    #     total_it_each_epoch = len(self.data_loaders[mode])
    #     dataloader_iter = iter(self.data_loaders[mode])
    #     with tqdm.trange(0, total_it_each_epoch, desc='eval_epochs', dynamic_ncols=True, leave=(self.args.local_rank == 0)) as pbar:
    #         for cur_it in pbar:
    #             try:
    #                 input_dict = next(dataloader_iter)
    #             except StopIteration:
    #                 dataloader_iter = iter(self.data_loaders['test'])
    #                 input_dict = next(dataloader_iter)
    #             output_dict = self.net(input_dict)
    #             batch_size, top, fde, t_ade, t_fde = self.net.module.compute_model_metrics(input_dict, output_dict)
    #             total += torch.tensor(batch_size).cuda()
    #             top_acc += torch.tensor(top).cuda()
    #             target_fde += torch.as_tensor(fde, device='cuda')
    #             traj_ade += torch.as_tensor(t_ade, device='cuda')
    #             traj_fde += torch.as_tensor(t_fde, device='cuda')
    #     dist.barrier()
    #     for x in [total, top_acc, target_fde, traj_ade, traj_fde]:
    #         dist.reduce(x, 0) # reduce并返回给进程0
  
    #     if self.args.local_rank == 0:
    #         top_acc = 100 * top_acc.item() / total.item()
    #         target_fde = target_fde.item() / total.item()
    #         traj_ade = traj_ade.item() / total.item()
    #         traj_fde = traj_fde.item() / total.item()
    #         self.logger.info(f'Eval: case_num={total.item()}, top_acc={top_acc:.2f}%, target_fde={target_fde:.5f}, traj_ade={traj_ade:.5f}, traj_fde={traj_fde:.5f}')
    #         if traj_ade < self.best_ade:
    #             self.best_ade = traj_ade
    #             self._save_checkpoint(epoch, best_epoch=True)  # save model
    #             self.logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        
if __name__ == "__main__":
    from parser_args import  main_parser
    args = main_parser()
    piple = Trainer(args)
    piple.train()
            

    