import torch
import torch
import torch.nn as nn
from model import DAVNet2D

def dice_loss(X, Y):
    return 2 * (torch.dot(X, Y) / (torch.dot(X, X) + torch.dot(Y, Y)))

default_configs = {
    'batch_size': 8,
    'learning_rate':  None,
    'seg_loss': 'dice',
    'domain_loss': 'bce',
    'print_progress': True,
    'model': 'davnet2d',
    'classes': 6,
    'half_precision': False,
    'cuda': True,
    'checkpoint_dir' = '~/davnet/checkpts'
}

models = {
    'davnet2d': DAVNet2D
}

losses = {
    'dice': utils.dice_loss,
    'bce': nn.BCELoss()
}

def identity_tracker(x, **kwargs):
    return x
