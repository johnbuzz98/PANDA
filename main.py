import numpy as np
import os
import random
import wandb

import torch
import argparse

import logging

from train import fit
from models import *
from datasets import get_loaders
from log import setup_default_logging
from copy import deepcopy

from losses import  EWCLoss
from utils import freeze_model

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(args):
    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # load model
    model = resnet.get_resnet_model(resnet_type=args.resnet_type)
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))
    # load dataloader
    trainloader, testloader = get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    # set training
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)

    #EWC?
    ewc_loss=None
    ses = False
    if args.pandatype == 'ewc':
        ewc = True
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    elif args.pandatype == 'es':
        args.epochs = 15
        ewc = False
    elif args.pandatype == 'ses':
        ewc = False
        ses = True
    else:
        raise NotImplementedError

    # initialize wandb
    wandb.init(name=args.exp_name, project='PANDA-WJ', config=args)
    
    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        ewc          = ewc,
        ewc_loss     = ewc_loss,
        optimizer    = optimizer, 
        epochs       = args.epochs,
        savedir      = savedir,
        log_interval = args.log_interval,
        ses          = ses,  
        device       = device)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PANDA TEST")

    parser.add_argument('--exp-name', type=str, default='PANDA', help='experiment name')
    parser.add_argument('--dataset', default='fashion',  help='cifar10 or fashion')
    parser.add_argument('--datadir', default='./data', help='dataset directory')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fisher matrix diagonal path')
    parser.add_argument('--savedir', default='./save', help='save directory')
    parser.add_argument('--pandatype', type=str, default='ses', help='ewc, es, ses')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet-type', default=152, type=int, help='which resnet to use')
    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=32,help='batch size')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')

    # seed
    parser.add_argument('--seed',type=int,default=216,help='216 is my birthday')

    args = parser.parse_args()

    run(args)