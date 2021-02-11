import torch
import torch
import torch.nn as nn
from model import DAVNet2D
import pickle
from torch.utils.data import DataLoader
import time
from threading import Thread

def batch_flatten(X):
    return X.view(X.size(0), -1)

def batch_and_class_flatten(X):
    return X.view(X.size(0), X.size(1), -1)

def dice_loss_normal(Y_hat, Y, smooth=1e-10):
    return (-torch.log(dice_score(Y_hat, Y, smooth))).sum(0)

def dice_score(Y_hat, Y, smooth=1e-10, flat=False):
    assert Y_hat.size() == Y.size()
    if not flat:
        Y, Y_hat = batch_flatten(Y), batch_flatten(Y_hat)
    intersection = (Y * Y_hat).sum(-1)
    union = Y.sum(-1) + Y_hat.sum(-1)
    return (2 * intersection + smooth) / (union + smooth)

def dice_loss_weighted(Y_hat, Y, exp=0.7, smooth=1e-10):
    assert Y_hat.size() == Y.size()
    background_sum = Y[:, 0, :, :].sum()
    for i in range(Y.size(1)):
        Y[:, i, :, :] = Y[:, i, :, :] * (safe_div(background_sum, Y[:, i, :, :].sum(), 1) ** exp)
    return dice_loss_normal(Y_hat, Y, smooth)

def per_class_dice(Y_hat, Y, tolist=True):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = batch_and_class_flatten(Y), batch_and_class_flatten(Y_hat)
    dice = 2 * ((Y * Y_hat).sum(-1) / (Y + Y_hat).sum(-1)).mean(0).squeeze()
    if tolist:
        dice = dice.tolist()
    return dice

def per_class_loss(Y_hat, Y, classes=default_configs['classes'], batch=default_configs['all_source_batch_size']):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = batch_and_class_flatten(Y), batch_and_class_flatten(Y_hat)
    dice = 2 * ((Y * Y_hat).sum(-1) / (Y + Y_hat).sum(-1)).sum()
    return 1 - dice / (classes * batch)

default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 32,
    'learning_rate':  10e-5,
    'seg_loss': 'per_class_loss',
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
    'blind_target': True,
    'warmup_length': 20,
    'checkpoint': None,
    'log_frequency': None,
    'MDD_sample_size': 10,
    'domain_loss_weight': 1,
    'disc_in': [3, 4, 5, 6]
}

models = {
    'davnet2d': DAVNet2D
}

losses = {
    'dice': dice_loss_normal,
    'weighted_dice': dice_loss_weighted,
    'bce': nn.NLLLoss(),
    'per_class_loss': per_class_loss
}

log = print

phase_counter = {
    'train': 0,
    'valid': 0
}

all_metrics = ['sample_count', 'balanced_sample_count', 'running_domain_loss',
    'running_domain_acc', 'running_seg_loss',
    'pred_source', 'pred_target',  'epoch_domain_loss', 'epoch_domain_acc', 'epoch_seg_loss',
    'running_per_class_loss', 'mean_discrepancy']

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
