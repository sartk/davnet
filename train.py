import gc
import time
import numpy as np
from tqdm import tqdm
from dataset import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from pprint import pprint

def train(**kwargs):

    configs = default_configs.copy()
    configs.update(kwargs)
    N = configs['num_epochs']

    tracker = tqdm if configs['print_progress'] else identity_tracker
    os.environ['CUDA_VISIBLE_DEVICES'] = configs['CUDA_VISIBLE_DEVICES']
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    n = configs['num_workers']
    dataloaders = {
        'balanced': {
            'train': DataLoader(dataset=kMRI('train', balanced=True, group='all'), batch_size=configs['balanced_batch_size'], shuffle=True),
            'valid': DataLoader(dataset=kMRI('valid', balanced=True, group='all'), batch_size=configs['balanced_batch_size'], shuffle=True)
        },
        'all_source': {
            'train': DataLoader(dataset=kMRI('train', balanced=False, group='source'), batch_size= configs['all_source_batch_size'], shuffle=True),
            'valid': DataLoader(dataset=kMRI('valid', balanced=False, group='source'), batch_size= configs['all_source_batch_size'], shuffle=True)
        }
    }

    model = models[configs['model']](classes=configs['classes'])
    data_type = torch.HalfTensor if configs['half_precision'] else torch.FloatTensor

    if configs['half_precision']:
        model = model.half()
    else:
        model = model.float()

    if configs['cuda']:
        model = model.cuda()

    F_seg_loss = losses[configs['seg_loss']]
    F_domain_loss = losses[configs['domain_loss']]

    if  configs['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'],
                               weight_decay=configs['weight_decay'])
    elif configs['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=configs['learning_rate'],
                              weight_decay=configs['weight_decay'], momentum=0.9)
    else:
        raise NotImplementedError('{} not setup.'.format(configs['optimizer']))

    if configs['checkpoint']:
        checkpoint = torch.load(configs['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_valid_loss = {'seg': np.inf, 'domain': np.inf}
    patience_counter = 0
    groups = ['balanced']

    for epoch in tracker(range(N), desc='epoch'):

        if epoch == configs['all_source_epoch']:
            groups.insert(0, 'all_source')

        M = {}

        for m in all_metrics:
            M[m] = phase_counter.copy()

        for phase in tracker(iter(configs['phases']), desc='phase'):
            for group in tracker(iter(groups), desc='group'):
                model.train(phase == 'train')
                dataloader = dataloaders[group][phase]
                len_dataloader = len(dataloader)
                i = 0
                for img, seg_label, domain_label in tracker(dataloader, desc='batch'):
                    p = float(i + epoch * len_dataloader) / N / len_dataloader
                    grad_reversal_coef = configs['grad_reversal_coef'] / (1. + np.exp(-configs['grad_reversal_growth'] * p)) - 1

                    if configs['cuda']:
                        img = img.cuda(non_blocking=True)
                        seg_label = seg_label.cuda(non_blocking=True)
                        domain_label = domain_label.cuda(non_blocking=True)

                    if configs['half_precision']:
                        img = img.half()
                        seg_label = seg_label.half()
                        domain_label = domain_label.half()

                    optimizer.zero_grad()

                    if group == 'balanced':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=False)
                    elif group == 'all_source':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=True), torch.tensor([[1, 0]] * configs['all_source_batch_size']).float().cuda()

                    is_source = (domain_label.argmax(1) == 0).int()
                    is_target = (domain_label.argmax(1) == 1).int()
                    M['labeled_source'][phase] += is_source.sum().item()
                    M['labeled_target'][phase] += is_target.sum().item()

                    # hide segmentation labels from target dataset
                    if configs['blind_target']:
                        seg_label = (is_source * seg_label) + (is_target * seg_pred)

                    seg_loss = F_seg_loss(seg_pred, seg_label)
                    domain_loss = F_domain_loss(domain_pred, domain_label)
                    err = (seg_loss + domain_loss)

                    if phase == 'train':
                        err.backward()
                        optimizer.step()

                    if group == 'balanced':
                        M['running_domain_acc'][phase] += (domain_pred.argmax(1) == domain_label.argmax(1)).sum().item()
                        M['balanced_sample_count'][phase] += img.size(0)

                    M['pred_source'][phase] += (domain_pred.argmax(1) == 0).sum().item()
                    M['pred_target'][phase] += (domain_pred.argmax(1) == 1).sum().item()
                    M['sample_count'][phase] += img.size(0)
                    M['running_seg_loss'][phase] += seg_loss.item()
                    M['running_domain_loss'][phase] += domain_loss.item() * img.size(0)
                    i += 1

                pprint(M)

            M['epoch_domain_loss'][phase] = M['running_domain_loss'][phase] / M['sample_count'][phase]
            M['epoch_domain_acc'][phase] = M['running_domain_acc'][phase] / M['balanced_sample_count'][phase]
            M['epoch_seg_loss'][phase] = M['running_seg_loss'][phase] / M['sample_count'][phase]

            pprint(M)

        if (M['epoch_domain_loss']['valid'] < best_valid_loss['domain']) or (M['epoch_seg_loss']['valid'] < best_valid_loss['seg']):
            patience_counter = 0
            best_valid_loss['domain'] = M['epoch_domain_loss']['valid']
            M['epoch_seg_loss']['valid'] = best_valid_loss['seg']
            with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}-{epoch}.pt'), 'wb+') as f:
                torch.save({
                            'epoch': epoch,
                            'phase': phase,
                            'groups': groups,
                            'metrics': M,
                            'configs': configs,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f)
        else:
            patience_counter += 1
            if patience_counter < configs['patience']:
                print('\nPatience counter {}/{}.'.format(patience_counter, configs['patience']))
            elif patience_counter == configs['patience']:
                print('\nEarly stopping. No improvement after {} Epochs.'.format(patience_counter))
                break

        gc.collect()
