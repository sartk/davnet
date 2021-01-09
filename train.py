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
import pdb
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
            groups.append('all_source')

        metrics = {'train': {}, 'valid': {}}
        for phase in metrics:
            for m in all_metrics:
                metrics[phase][m] = 0

        for phase in tracker(iter(configs['phases']), desc='phase'):
            M = metrics[phase]
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
                        domain_label = domain_label.argmax(-1).cuda(non_blocking=True)

                    if configs['half_precision']:
                        img = img.half()
                        seg_label = seg_label.half()

                    optimizer.zero_grad()

                    if group == 'balanced':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=False)
                    elif group == 'all_source':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=True), torch.tensor([[1, 0]] * img.size(0)).long().cuda()

                    is_source = (domain_label == 0).int()
                    is_target = (domain_label == 1).int()
                    M['labeled_source'] += is_source.sum().item()
                    M['labeled_target'] += is_target.sum().item()

                    #pdb.set_trace()
                    # hide segmentation labels from target dataset
                    if configs['blind_target']:
                        seg_label = (is_source * seg_label) + (is_target * seg_pred)

                    seg_loss, per_class_loss = F_seg_loss(seg_pred, seg_label)

                    if configs['blind_target']:
                        seg_loss = seg_loss * img.size(0) / is_source.sum()

                    domain_loss = F_domain_loss(domain_pred, domain_label)
                    err = (seg_loss + domain_loss)

                    if phase == 'train':
                        err.backward()
                        optimizer.step()

                    if group == 'balanced':
                        M['running_domain_acc'] += (domain_pred.argmax(1) == domain_label).sum().item()
                        M['balanced_sample_count'] += img.size(0)

                    M['pred_source'] += (domain_pred.argmax(1) == 0).sum().item()
                    M['pred_target'] += (domain_pred.argmax(1) == 1).sum().item()
                    M['sample_count'] += img.size(0)
                    M['running_seg_loss'] += seg_loss.item()
                    M['running_domain_loss'] += domain_loss.item() * img.size(0)
                    M['running_per_class_loss'] += per_class_loss
                    i += 1

                    if i % 100 == 0:
                        N = M.copy()
                        N['running_per_class_loss'] = N['running_per_class_loss']/N['sample_count']
                        N['running_domain_acc'] = N['running_domain_acc']/N['sample_count']
                        N['running_domain_loss'] = N['running_domain_loss']/N['sample_count']
                        N['running_seg_loss'] = N['running_seg_loss']/N['sample_count']
                        pprint(N)

            M['epoch_domain_loss'] = M['running_domain_loss'] / M['sample_count']
            M['epoch_domain_acc'] = M['running_domain_acc'] / M['balanced_sample_count']
            M['epoch_seg_loss'] = M['running_seg_loss'] / M['sample_count']
            M['running_seg_loss'] = M['running_seg_loss']/M['sample_count']

            pprint(M)

        if (metrics['valid']['epoch_domain_loss'] < best_valid_loss['domain']) or (metrics['valid']['epoch_seg_loss'] < best_valid_loss['seg']):
            patience_counter = 0
            best_valid_loss['domain'] = metrics['valid']['epoch_domain_loss']
            metrics['valid']['epoch_seg_loss'] = best_valid_loss['seg']
            with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}-{epoch}.pt'), 'wb+') as f:
                torch.save({
                            'epoch': epoch,
                            'phase': phase,
                            'groups': groups,
                            'metrics': metrics,
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
