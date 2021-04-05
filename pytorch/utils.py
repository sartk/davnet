import torch
import torch
import torch.nn as nn
from model import DAVNet2D
import pickle
from torch.utils.data import DataLoader
import time
from threading import Thread
from dataset import kMRI
from configs import *
from scipy.spatial.distance import dice as python_dice

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

def batch_flatten(X):
    return X.view(X.size(0), -1)

def batch_and_class_flatten(X):
    return X.view(X.size(0), X.size(1), -1)

def dice_loss_normal(Y_hat, Y, smooth=1e-10):
    return (-torch.log(dice_score(Y_hat, Y, smooth))).sum(0)

def dice_score(Y_hat, Y, smooth=1e-10, flat=False, p=2):
    assert Y_hat.size() == Y.size()
    if not flat:
        Y, Y_hat = batch_flatten(Y), batch_flatten(Y_hat)
    intersection = (Y * Y_hat).sum(-1)
    union = (Y * Y).sum(-1) + (Y_hat.pow(p) * Y_hat.pow(p)).sum(-1)
    return (2 * intersection + smooth) / (union + smooth)

def dice_loss_weighted(Y_hat, Y, exp=0.7, smooth=1e-10):
    assert Y_hat.size() == Y.size()
    background_sum = Y[:, 0, :, :].sum()
    for i in range(Y.size(1)):
        Y[:, i, :, :] = Y[:, i, :, :] * (safe_div(background_sum, Y[:, i, :, :].sum(), 1) ** exp)
    return dice_loss_normal(Y_hat, Y)

def per_class_dice(Y_hat, Y, tolist=True, p=2, repr=''):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = batch_and_class_flatten(Y), batch_and_class_flatten(Y_hat) # [N, C, flat]
    dice = 2 * (((Y * Y_hat).sum(-1)) / (Y.pow(p).sum(-1) + Y_hat.pow(p).sum(-1))).mean(0).squeeze()
    if repr == '-log':
        dice = -torch.log(dice)
    elif repr == '1-':
        dice = 1 - dice
    if tolist:
        dice = dice.tolist()
    return dice

def native_per_class(Y_hat, Y):
    return per_class_dice(Y_hat, Y, False, 2, '1-').sum()

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


def logger(timestamp, delim=','):
    def log(*args):
        line = delim.join(map(str, args))
        with open(f'/data/bigbone6/skamat/checkpoints-davnet/logs/{timestamp}.log', 'a+') as f:
            f.write(line)
            f.write('\n')
    return log

source_ds = kMRI('valid', balanced=False, group='source')
target_ds = kMRI('valid', balanced=False, group='target')
#per_class_dice = DiceLoss(repr='')

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
            new_dice = per_class_dice(model(img, seg_only=True), seg, tolist=True)
            dice[group] = [d + n for d, n in zip(dice[group], new_dice)]

    return [d / batches for d in dice['source']], [d / batches for d in dice['target']]
