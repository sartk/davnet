import torch
import torch
import torch.nn as nn
from model import DAVNet2D
import numpy as np

def dice_loss(Y_hat, Y):
    eps = 0.000001
    _, result_ = Y_hat.max(1)
    result_ = torch.squeeze(result_)
    if Y_hat.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        Y_ = torch.cuda.FloatTensor(Y.size())
    else:
        result = torch.FloatTensor(result_.size())
        Y_ = torch.FloatTensor(Y.size())
    result.copy_(result_.data)
    Y_.copy_(Y.data)
    Y = Y_
    intersect = torch.dot(result, Y)

    result_sum = torch.sum(result)
    Y_sum = torch.sum(Y)
    union = result_sum + Y_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the Y volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    #    print('union: {:.3f}\t intersect: {:.6f}\t Y_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    #        union, intersect, Y_sum, result_sum, 2*IoU))
    return 2 * IoU

default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 16,
    'learning_rate':  10e-5,
    'seg_loss': 'dice',
    'domain_loss': 'bce',
    'weight_decay': 1,
    'print_progress': True,
    'model': 'davnet2d',
    'classes': 6,
    'half_precision': False,
    'cuda': True,
    'num_epochs': 100,
    'checkpoint_dir': '~/davnet-checkpts',
    'groups': ['balanced', 'all_source'],
    'phases': ['train', 'valid'],
    'num_workers': 4,
    'optimizer': 'adam'
}

models = {
    'davnet2d': DAVNet2D
}

losses = {
    'dice': dice_loss,
    'bce': nn.BCELoss()
}

def identity_tracker(x, **kwargs):
    return x
