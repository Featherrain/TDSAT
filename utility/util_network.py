import os
import torch.nn.init as init
from models.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .loss_functions import *
from tensorboardX import SummaryWriter
import socket
from datetime import datetime

def init_params(net, init_type='kn'):
    print('use init scheme: %s' % init_type)
    if init_type != 'edsr':
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                if init_type == 'kn':
                    init.kaiming_normal_(m.weight, mode='fan_out')
                if init_type == 'ku':
                    init.kaiming_uniform_(m.weight, mode='fan_out')
                if init_type == 'xn':
                    init.xavier_normal_(m.weight)
                if init_type == 'xu':
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def init_criterion(loss):
    if loss == 'l2':
        criterion = nn.MSELoss()
    elif loss == 'l1':
        criterion = nn.L1Loss()
    elif loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss == 'ssim':
        criterion = SSIMLoss(data_range=1, channel=31)
    elif loss == 'l2_ssim':
        criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
    else:
        criterion = nn.MSELoss()
    return criterion

def get_summary_writer(log_dir, prefix=None):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if prefix is None:
        log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
    else:
        log_dir = os.path.join(log_dir, prefix+'_'+datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer

def adjust_learning_rate(optimizer, lr):
    print('Adjust Learning Rate => %.4e' %lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def display_learning_rate(optimizer):
    lrs = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        print('learning rate of group %d: %.4e' % (i, lr))
        lrs.append(lr)
    return lrs