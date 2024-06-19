import datetime
import random
import torch
import time
import numpy as np
import logging
import torch.distributed as dist
import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '32153'
# os.environ['RANK'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['WORLD_SIZE'] = '1'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def init_dist_pytorch(local_rank, backend='nccl'):
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend
    )
    rank = dist.get_rank()
    return num_gpus, rank

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def formatted_time(elapsed_time):
    """
    Given an elapsed time in seconds, return a string with a nice format
    """
    if elapsed_time >= 3600:
        return str(datetime.timedelta(seconds=int(elapsed_time)))
    elif elapsed_time >= 60:
        minutes, seconds = elapsed_time // 60, elapsed_time % 60
        return f"{minutes:.0f} min and {seconds:.0f} sec"
    else:
        return f"{elapsed_time:.2f} sec"
    
def add_dict_prefix(d_in: dict, prefix: str):
    """
    Add prefix sub string to a dictionary with string keys
    """
    d_out = {prefix + '_' + k: v for k, v in d_in.items()}
    return d_out


def timed_main(function):
    """
    主函数的装饰器，计算时间的差值;
    """
    def decorator(*args, **kw):
        start = time.time()
        now = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
        print("Program started", now)
        result = function(*args, **kw)
        now = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
        elapsed_time = time.time() - start
        print('\nProgram finished {}. Elapsed time: {}'
              .format(now, formatted_time(elapsed_time)))
        return result
    return decorator

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def set_seed(seed_value):
    """
    Set random, numpy, torch and cuda seeds for reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """
    Given a PyTorch model, count all parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """
    Given a PyTorch model, count only trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(net, model_name):
    """
    Print network name and parameters
    """
    print(f"Model {model_name} is ready!")
    print("Total number of parameters:", count_parameters(net))
    print("Number of trainable parameters:", count_trainable_parameters(net))
    print(f'Number of trainable parameters:{count_trainable_parameters(net)/(1024*1024):.2f}M')
    print(f"Number of non-trainable parameters: "
          f"{count_parameters(net) - count_trainable_parameters(net)}\n")
            
def find_trainable_layers(model, verbose=False):
    """
    Given a PyTorch model, returns trainable layers
    """
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            if verbose:
                print(name)
    return params_to_update

# 识别障碍物当前的行为
def behavior_recognition(lane, lane_heading, base_point, base_heading, is_reverse):
    delta_yaw = normalize_angle(base_heading - lane_heading)
    behavior = 'INVALID'
    # 判断是左转还是右转
    if abs (delta_yaw) >= math.pi/6:
        behavior = 'TURN_RIGHT' if delta_yaw > 0 else 'TURN_LEFT'
    # 判断直行、左变道or右边道
    else:
        s, d = lane.xy_to_sd(base_point[0], base_point[1])
        if (d > 0 and not is_reverse) or (d < 0 and is_reverse):
            lane_width = lane.left_width_curve().calc(s)
        else:
            lane_width = lane.right_width_curve().calc(s)
        if abs(d) < lane_width:
            behavior = 'GO_STRAIGHT'
        elif (d > 0 and not is_reverse) or (d < 0 and is_reverse):
            behavior = 'CHANGE_RIGHT'
        else:
            behavior = 'CHANGE_LEFT'
        # print(f'lane_id: {lane.id}')
        # print(f'lane_width: {lane_width}')
        # print(f'd: {d}')
        # print(f'base_point: {base_point}')
        # print(behavior)
        # print("##"*12)
    return behavior
