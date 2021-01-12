import torch
import torch
import torch.nn as nn
from model import DAVNet2D
import pickle
def batch_flatten(X):
    return X.view(X.size(0), -1)

def dice_loss(Y_hat, Y, smooth=1e-10, save=False):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = batch_flatten(Y), batch_flatten(Y_hat)
    intersection = (Y * Y_hat).sum(1)
    union = Y.sum(1) + Y_hat.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return -torch.log(dice)

default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 32,
    'learning_rate':  10e-5,
    'seg_loss': 'dice',
    'domain_loss': 'bce',
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
    'blind_target': False,
    'all_source_epoch': 20,
    'checkpoint': None,
    'log_frequency': 100
}

models = {
    'davnet2d': DAVNet2D
}

losses = {
    'dice': dice_loss,
    'bce': nn.NLLLoss()
}

log = print

phase_counter = {
    'train': 0,
    'valid': 0
}

all_metrics = ['sample_count', 'balanced_sample_count', 'running_domain_loss',
    'running_domain_acc', 'running_seg_loss', 'labeled_source', 'labeled_target',
    'pred_source', 'pred_target',  'epoch_domain_loss', 'epoch_domain_acc', 'epoch_seg_loss',
    'running_per_class_loss']

def identity_tracker(x, **kwargs):
    return x
