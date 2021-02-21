import torch
import torch
import torch.nn as nn
from model import DAVNet2D
import pickle
from torch.utils.data import DataLoader
import time
from threading import Thread
from dataset import kMRI

default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 16,
    'learning_rate':  10e-5,
    'seg_loss': 'dice',
    'domain_loss': 'nll',
    'weight_decay': 1,
    'print_progress': True,
    'model': 'davnet2d',
    'classes': 4,
    'half_precision': False,
    'cuda': True,
    'num_epochs': 100,
    'checkpoint_dir': '/data/bigbone6/skamat/checkpoints-davnet',
    'plots_dir': '/data/bigbone6/skamat/plots-davnet',
    'phases': ['train', 'valid'],
    'num_workers': 4,
    'optimizer': 'sgd',
    'plot_progress': True,
    'patience': 5,
    'grad_reversal_coef': 2,
    'grad_reversal_growth': 10,
    'blind_target': True,
    'warmup_length': 20,
    'checkpoint': None,
    'log_frequency': 100,
    'MDD_sample_size': 10,
    'domain_loss_weight': 1,
    'disc_in': [3, 4, 5, 6],
    'valid_freq': 1000,
    'message': '',
    'dice_weights': None
}

phase_counter = {
    'train': 0,
    'valid': 0
}

all_metrics = ['sample_count', 'balanced_sample_count', 'running_domain_loss',
    'running_domain_acc', 'running_seg_loss',
    'pred_source', 'pred_target',  'epoch_domain_loss', 'epoch_domain_acc', 'epoch_seg_loss',
    'running_per_class_loss', 'mean_discrepancy']


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#github: @hubutui
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

#github: @hubutui
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=0.1, p=2, reduction='mean', representation='-log'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        if representation == '1-':
            loss = 1 - num / den
        elif representation == '-log':
            loss = -torch.log(num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

#github: @hubutui
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        #predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def identity_tracker(x, **kwargs):
    return x

def safe_div(x, y, default=0):
    return default if y == 0 else x / y

def random_sample(dataset, N, cuda=False):

    img, seg, _ = next(iter(DataLoader(dataset=dataset, batch_size=N, shuffle=True)))
    if cuda:
        img, seg = img.cuda(), seg.cuda()
    return img, seg

#from inputimeout import inputimeout, TimeoutOccurred

def update_hyper_param(configs):
    while needs_update():
        param = input("Which parameter >>>")
        new_value = input("Enter new value >>>")
        eval("configs['{}'] = {}".format(param, new_value))

def needs_update():
    try:
        return inputimeout(prompt='Need to update hyper params?', timeout=10).lower()[0] == 'y'
    except TimeoutOccurred:
        return False


models = {
    'davnet2d': DAVNet2D
}

losses = {
    'nll': nn.NLLLoss(),
}

def logger(timestamp, delim=','):
    def log(*args):
        line = delim.join(map(str, args))
        with open(f'/data/bigbone6/skamat/checkpoints-davnet/logs/{timestamp}.log', 'a+') as f:
            f.write(line)
            f.write('\n')
    return log

source_ds = kMRI('valid', balanced=False, group='source')
target_ds = kMRI('valid', balanced=False, group='target')

def baseline(N, model, cuda=True, num_classes=4):

    batches = N // 16
    dl = {
        'source': iter(DataLoader(dataset=source_ds, batch_size=16, shuffle=True)),
        'target': iter(DataLoader(dataset=target_ds, batch_size=16, shuffle=True))
    }
    dice = {
        'source': [0] * num_classes,
        'target': [0] * num_classes
    }

    for group in ['source', 'target']:
        for i in range(batches):
            img, seg, _ = next(dl[group])
            if cuda:
                img, seg = img.cuda(), seg.cuda()
            new_dice = per_class_dice(model(img, seg_only=True), seg,
                tolist=True)
            dice[group] = [d + n for d, n in zip(dice[group], new_dice)]

    return [d / batches for d in dice['source']], [d / batches for d in dice['target']]
