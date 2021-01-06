import torch
import torch
import torch.nn as nn
from model import DAVNet2D

def dice_loss(Y_hat, Y):
    assert Y_hat.size() == Y.size()
    Y, Y_hat = torch.flatten(Y, start_dim=1).double(), torch.flatten(Y_hat, start_dim=1).double()
    b, L = Y.size(0), Y.size(1)
    M1, M2 = lambda Y: Y.view(b, 1, L), lambda Y: Y.view(b, L, 1)
    I = 2 * torch.bmm(M1(Y), M2(Y_hat))
    U = torch.bmm(M1(Y), M2(Y)) + torch.bmm(M1(Y_hat), M2(Y_hat))
    return (I/U).float()

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
