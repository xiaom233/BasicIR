import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import copy
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


#### options
opt = parse_options(is_train=False)

make_exp_dirs(opt)
log_file = osp.join(opt['path']['log'],
                    f"test_{opt['name']}_{get_time_str()}.log")
logger = get_root_logger(
    logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
logger.info(get_env_info())
logger.info(dict2str(opt))
tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['root'], 'tb_logger', opt['name']))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    dataset_opt['phase'] = 'test'
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(
        test_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])
    logger.info(
        f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    test_loaders.append(test_loader)

model = create_model(opt)

opt_dir = []
for i in range(5000, 600000, 5000):
    temp_opt = copy.deepcopy(opt)
    previous_pth_path = temp_opt['path']['pretrain_network_g'].split('/models/')
    current_pth_path = previous_pth_path[0] + '/models/net_g_' + str(i) + '.pth'
    temp_opt['path']['pretrain_network_g'] = current_pth_path
    opt_dir.append(temp_opt)

for current_opt in opt_dir:
    model.load_network(model.net_g, current_opt['path']['pretrain_network_g'])
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        current_iteration_model = current_opt['path']['pretrain_network_g'].split('/models/')
        current_iteration = current_iteration_model[1]
        temp = current_iteration.replace('net_g_', '')
        temp = temp.replace('.pth', '')
        itration_str = temp
        itration_num = int(itration_str)
        print(itration_num)
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.test_all(
            test_loader,
            current_iter=itration_num,
            tb_logger=tb_logger,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)

