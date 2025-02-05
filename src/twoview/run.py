import argparse
import collections
import itertools
import torch
import os, random
import numpy as np

from model import mvc
from utils import cal_std, get_logger, drop_feature
from datasets import load_data
from configure import get_default_config

import tqdm
import ipdb
import time



dataset = {
0: "Caltech101-20",
1: "Scene_15",
2: "LandUse_21",
3: "NoisyMNIST",
            }

parser  =  argparse.ArgumentParser()
parser.add_argument('--dataset', type = int, default = '0', help = 'dataset id')
parser.add_argument('--devices', type = str, default = '0', help = 'gpu device ids')
parser.add_argument('--print_num', type = int, default = '10', help = 'gap of print evaluations')
parser.add_argument('--test_time', type = int, default = '5', help = 'number of test times')
parser.add_argument('--out', default='./log',help='directory to output the result')

parser.add_argument('--alpha', type = float, default = 1, help = 'trade-off hyperparameter of entropy contrastive loss conditional entropy')
parser.add_argument('--beta', type = float, default = 1, help = 'trade-off hyperparameter of entropy contrastive loss  entropy')
parser.add_argument('--gamma', type = float, default = 1, help = 'trade-off hyperparameter of entropy contrastive loss  entropy')

parser.add_argument('--lambda1', type = float, default = 0.1, help = 'trade-off hyperparameter of L_glb loss')
parser.add_argument('--lambda2', type = float, default = 0.1, help = 'trade-off hyperparameter of L_code loss')
parser.add_argument('--droprate', type = float, default = 0.1, help = 'hyperparameter of feature augmentation')

parser.add_argument('--data_seed', type = int, default = 2, help = 'hyperparameter of seed')

args  =  parser.parse_args()
dataset  =  dataset[args.dataset]


def main():
    # Environments
    # os.environ["CUDA_DEVICE_ORDER"]  =  "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]  =  str(args.devices)
    use_cuda = torch.cuda.is_available()
    # device = torch.device(args.devices if use_cuda else 'cpu')
    device = torch.device('cuda:{}'.format(args.devices) )
    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    config['training']['alpha'] = args.alpha
    config['training']['beta'] = args.beta
    config['training']['gamma'] = args.gamma
    config['training']['droprate'] = args.droprate
    config['training']['lambda1'] = args.lambda1
    config['training']['lambda2'] = args.lambda2
    # ipdb.set_trace()


    # Create a log file
    log_path =args.out
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    logger = get_logger(os.path.join(log_path, dataset +'.log'))
    logger.info('Dataset:' + str(dataset))

    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, gt = load_data(config)


    accumulated_metrics = collections.defaultdict(list)
    time_temp= time.time()
    for data_seed in range(1, args.test_time + 1):
        # Set random seeds
        seed = config['training']['seed']
        np.random.seed(data_seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4) 
        torch.backends.cudnn.deterministic = True
        # Load view1 and view2 with tensor
        X1, X2 = torch.from_numpy(X_list[0]).float().cuda(), torch.from_numpy(X_list[1]).float().cuda()
        

        # Build the model
        model = mvc(args, config)
        model.model_init()
        optimizer = torch.optim.Adam(
            itertools.chain(model.autoencoder1.parameters(), model.autoencoder2.parameters()),
            lr = config['training']['lr'])
        model.to_device(device)

        # Print the models
        logger.info(model.autoencoder1)
        logger.info(optimizer)

        # Training
        acc, nmi, ari = model.train(config, logger, X1, X2, gt, optimizer, device)

        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)


    logger.info('--------------------Training over--------------------')
    time_total=time.time()-time_temp
    logger.info('The total of training time is{}'.format(time_total))
    cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])

if __name__ == '__main__':
    main()