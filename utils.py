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
    def __init__(self, smooth=1e-9, p=2, reduction='mean', repr='-log'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.repr = repr

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match, {} and {}".format(predict.shape, target.shape)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        if self.repr == '1-':
            loss = 1 - num / den
        elif self.repr == '-log':
            loss = -torch.log(num / den)
        else:
            loss = num / den

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
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, per_class=False):
        assert predict.shape == target.shape, "predict & target batch size don't match, {} and {}".format(predict.shape, target.shape)
        #pdb.set_trace()
        if per_class:
            total_loss = [0] * target.shape[1]
        else:
            total_loss = 0

        #predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index or per_class:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                if per_class:
                    total_loss[i] = dice_loss.item()
                else:
                    total_loss += dice_loss

        if per_class:
            return total_loss
        else:
            return total_loss/target.shape[1]

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

def dice_loss_weighted(Y_hat, Y, exp=0.5, smooth=1e-10):
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

def dice_loss_fra(target,prediction,p=2,smooth=1e-9,return_mean = False):
    ncl = target.shape[1]
    per_class = np.zeros(ncl)
    for cl in range(ncl):
        pred = prediction[:,cl]
        targ = target[:,cl]
        inters = pred * targ
        numerator = (2 * torch.sum(inters,dim=(0,1,2)))
        denominator_sq = (torch.sum(targ**2,dim=(0,1,2))) + (torch.sum(pred**2,dim=(1,2)))
        per_class[cl] = numerator/(denominator_sq + smooth)
    if return_mean:
        return torch.mean(per_class), per_class
    else:
        return per_class

def py_dice(target, prediction):
    assert target.size() == prediction.size()
    total_across_batches = [0] * target.size(1)
    for b in range(target.size(0)):
        for c in range(target.size(1)):
            Y_hat = target[b, c, :, :].squeeze()
            Y = prediction[b, c, :, :].squeeze()
            total_across_batches[c] += 1.-torch.mean((2*(Y_hat*Y)).sum(-1).sum(-1)/(Y_hat.pow(2).sum(-1).sum(-1)+Y.pow(2).sum(-1).sum(-1)))
    return [b / target.size(0) for b in total_across_batches]

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
    'dice': dice_loss_normal,
    'weighted_dice': dice_loss_weighted,
    'nll': nn.NLLLoss(),
    'per_class': DiceLoss(),
    'native_per_class': per_class_dice
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
