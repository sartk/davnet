import torch
import torch
import torch.nn as nn
from model import DAVNet2D

def dice_loss(X, Y):
    return 2 * (torch.dot(X, Y) / (torch.dot(X, X) + torch.dot(Y, Y)))

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
