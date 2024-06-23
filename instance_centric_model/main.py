import os,sys
parent_dir = "/data/wangchen/instance_centric/"
sys.path.append(parent_dir)
import datetime

from utils import timed_main, set_seed, init_dist_pytorch, create_logger
from parser_args import  main_parser, log_args_to_file
from src.trainer import Trainer

def main():
    args = main_parser()
    assert args.use_cuda == True, 'GPU des not support!!'
    total_gpus, args.local_rank = init_dist_pytorch(args.local_rank, backend='nccl')
    if args.reproducibility:
        set_seed(seed_value=7777)
        
    log_file = args.log_dir + "/log_train_%s.txt"%datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file, args.local_rank)
    
    # log to file
    logger.info('**********************Start logging**********************')
    log_args_to_file(args, logger)
    processor = Trainer(args, logger)
    processor.train()
    logger.info('**********************End training**********************')
if __name__ == '__main__':
    main()