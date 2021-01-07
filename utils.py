import torch
import torch
import torch.nn as nn
from model import DAVNet2D

def batch_flatten(X):
    return X.view(X.size(0), -1)

def dice_loss(Y_hat, Y, smooth=1e-10):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = batch_flatten(Y), batch_flatten(Y_hat)
    intersection = (Y * Y_hat).sum(1)
    union = Y.sum(1) + Y_hat.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).sum()

default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 32,
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
    'checkpoint_dir': '/data/bigbone6/skamat/checkpoints-davnet',
    'plots_dir': '/data/bigbone6/skamat/plots-davnet',
    'groups': ['balanced', 'all_source'],
    'phases': ['train', 'valid'],
    'num_workers': 4,
    'optimizer': 'adam',
    'plot_progress': True,
    'patience': 5,
    'grad_reversal_coef': 8,
    'grad_reversal_growth': 10,
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
