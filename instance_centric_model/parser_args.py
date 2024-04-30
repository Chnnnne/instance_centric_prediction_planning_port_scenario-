import argparse
import torch
import os
import yaml
from utils import str2bool

# 训练使用的parser
def get_parser():
    parser = argparse.ArgumentParser(description="arg parser")
    
    ##############################################################
    # Dataset, test set and model
    ##############################################################
    parser.add_argument(
        '--model_name', '-mn', default='MODEL', type=str,
        choices=['MODEL'], help='Type of architecture to use')
    parser.add_argument(
        '--dataset_dir', '-dd', default=f'/private/wanggang/instance_centric_data/', type=str,
        help='Set the parent dir of train data and valid data')
    parser.add_argument(
        '--batch_size', '-bs', default=8, type=int, help='number of batch size')
    parser.add_argument(
        '--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument(
        '--enable_log', '-el', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to open the tensorboard log")
    parser.add_argument(
        '--device', default="cuda", type=str, help='What device to use')
    ##############################################################
    # Distribu
    ##############################################################
    parser.add_argument(
        '--local_rank',  type=int, default=0, help='local_rank')
    parser.add_argument(
        '--without_sync_bn', default=False, type=str2bool, const=False,
        nargs='?', help="without_sync_bn")
    
    ##############################################################
    # Training/testing parameters
    ##############################################################
    parser.add_argument(
        '--phase', '-ph', default='train', type=str,
        choices=['train', 'test'],
        help='Phase selection. During test phase you need to load a '
             'pre-trained model')
    parser.add_argument(
        '--num_epochs', '-ne', default=50, type=int, help='number of epochs to train for')
    parser.add_argument(
        '--load_checkpoint', '-lc', default=None, type=str,
        help="Load pre-trained model for testing or resume training. Specify "
             "the epoch to load or 'best' to load the best model. Default=None "
             "means do not load any model.")
       
    parser.add_argument(
        '--save_every', '-se', default=10, type=int,
        help="Save model weights and outputs every save_every epochs.")
    
    parser.add_argument(
        '--start_validation', default=10, type=int,
        help="Validate the model starting from this epoch")
    # 1 for slow but accurate results, 20 for fast results
    parser.add_argument(
        '--validate_every', '-ve', default=10, type=int,
        help="Validate model every validate_every epochs")
    
    ##############################################################
    # Deep Learning strategies
    ##############################################################
    parser.add_argument(
        '--learning_rate', '-lr', default=1e-3, type=float)
    parser.add_argument(
        '--weight_decay', '-wd', default=0.01, type=float)
    parser.add_argument(
        '--optimizer', default='AdamW', type=str, choices=['Adam', 'SGD', 'AdamW'],
        help='Optimizer selection')
    parser.add_argument(
        '--scheduler', default='ExponentialLR', type=str, choices=[
            'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', None],
        help='Learning rate scheduler')
    
    ##############################################################
    # Model strategies
    ##############################################################
    parser.add_argument(
        '--agent_input_size', default=13, type=int)
    parser.add_argument(
        '--agent_hidden_size', default=64, type=int)
    parser.add_argument(
        '--map_input_size', default=5, type=int)
    parser.add_argument(
        '--map_hidden_size', default=64, type=int)
    parser.add_argument(
        '--d_model', default=128, type=int)

    parser.add_argument(
        '--rpe_input_size', default=5, type=int)
    parser.add_argument(
        '--rpe_hidden_size', default=64, type=int)
    parser.add_argument(
        '--plan_input_size', default=4, type=int)
    parser.add_argument(
        '--decoder_hidden_size', default=64, type=int)
    parser.add_argument(
        '--bezier_order', default=7, type=int)
    parser.add_argument(
        '--dropout', default=0.1, type=float)
    parser.add_argument(
        '--m', default=50, type=int)
    
    parser.add_argument('--update_edge', default=True, type=bool)
    parser.add_argument('--init_weights', default=True, type=bool)

    ##############################################################
    # Debug parameters
    ##############################################################
    
    parser.add_argument(
        '--reproducibility', '-r', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to set the seed for reproducibility")
    return parser
    
    
def check_and_add_additional_args(args):
    """
    Add default paths, device and other additional args to parsed args
    """
    # 根据实际情况设置GPU或者CPU
    if args.device.startswith('cuda') and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.device = 'cpu'
        args.use_cuda = False
        
    # 定义各种保存路径
    args.base_dir = '.'          
    args.save_base_dir = 'output' # 用于保存输出和模型来的目录
    args.save_dir = os.path.join(args.base_dir, args.save_base_dir)
    args.model_dir = os.path.join(args.save_dir, args.model_name)
    args.log_dir = os.path.join(args.save_dir, 'log')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    args.config = os.path.join(args.save_dir, 'config_' + args.model_name + '.yaml')
    return args
    
def save_args(args):
    """
    将参数保存到config.yaml文件
    """
    args_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)
        
def load_args(cur_parser, parsed_args):
    """
    从保存的config文件中加载参数，
    The priority is:
    command line > saved configuration files > default values in script.
    """
    with open(parsed_args.config, 'r') as f:
        saved_args = yaml.full_load(f)
    for k in saved_args.keys():
        if k not in vars(parsed_args).keys():
            raise KeyError('WRONG ARG: {}'.format(k))
    assert set(saved_args) == set(vars(parsed_args)), \
        "Entered args and config saved args are different"
    cur_parser.set_defaults(**saved_args)
    return cur_parser.parse_args([])

def print_args(args):
    print("-"*62)
    print("|" + " "*25 + "PARAMETERS" + " "*25 + "|")
    print("-" * 62)
    for k, v in vars(args).items():
        print(f"| {k:25s}: {v}")
    print("-"*62 + "\n")

    
def log_args_to_file(args, logger=None):
    logger.info("-"*62)
    logger.info("|" + " "*25 + "PARAMETERS" + " "*25 + "|")
    logger.info("-" * 62)
    for k, v in vars(args).items():
        logger.info(f"| {k:25s}: {v}")
    logger.info("-"*62 + "\n")
    

def main_parser():
    parser = get_parser()
    parsed_args = parser.parse_args([])
    parsed_args = check_and_add_additional_args(parsed_args)

    # TODO(wg) 由于多进程的缘故，保存yaml和读取yaml会存在问题
    # # 加载config的yaml文件
    # if os.path.exists(parsed_args.config):
    #     print(parsed_args.config)
    #     parsed_args = load_args(parser, parsed_args)
    # else:
    #     save_args(parsed_args)
    save_args(parsed_args)
        
    # print args given to the model
    return parsed_args

    
