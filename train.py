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
from itertools import cycle
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

    model = models[configs['model']](classes=configs['classes'], disc_in=configs['disc_in'])
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
    warmup = True

    for epoch in tracker(range(N), desc='epoch'):

        if epoch == configs['warmup_length']:
            warmup = False

        metrics = {'train': {}, 'valid': {}}

        for phase in metrics:

            for m in all_metrics:
                metrics[phase][m] = 0

            M = metrics[phase]
            model.train(phase == 'train')
            i = 0
            a = dataloaders['all_source'][phase]
            len_dataloader = len(a)
            b = cycle([(None, None, None)] if warmup else dataloaders['balanced'][phase])
            iterator = tracker(zip(a, b), desc='batch')

            for ((img_a, seg_label, _), (img_b, _, dlab)) in iterator:

                n = img_a.size(0)
                p = float(i + epoch * len_dataloader) / N / len_dataloader
                grad_reversal_coef = configs['grad_reversal_coef'] / (1. + np.exp(-configs['grad_reversal_growth'] * p)) - 1

                if configs['cuda']:
                    img = img_a.cuda(non_blocking=True)
                    seg_label = seg_label.cuda(non_blocking=True)
                    if not warmup:
                        img_b = img_b.cuda(non_blocking=True)
                        domain_label = dlab.cuda(non_blocking=True).argmax(-1).detach()

                if configs['half_precision']:
                    img = img.half()
                    img_b = img_b.half()
                    seg_label = seg_label.half()

                optimizer.zero_grad()

                seg_pred = model(img, grad_reversal_coef, seg_only=True)
                seg_loss = F_seg_loss(seg_pred, seg_label)

                if warmup:
                    err = seg_loss
                else:
                    _, domain_pred = model(img_b, grad_reversal_coef, seg_only=False)
                    domain_loss = F_domain_loss(domain_pred, domain_label)
                    M['running_domain_acc'] += (domain_pred.argmax(1) == domain_label).sum().item()
                    M['balanced_sample_count'] += n
                    M['running_domain_loss'] += (domain_loss * n).item()
                    M['pred_source'] += (domain_pred.argmax(1) == 0).sum().item()
                    M['pred_target'] += (domain_pred.argmax(1) == 1).sum().item()
                    err = seg_loss + domain_loss

                if phase == 'train':
                    err.backward()
                    optimizer.step()

                M['running_seg_loss'] += seg_loss.item()
                M['sample_count'] += n
                i += 1

                if configs['log_frequency'] and i % configs['log_frequency'] == 0:
                    log('Domain Loss',  safe_div(M['running_domain_loss'], M['balanced_sample_count']))
                    log('Domain Acc', safe_div(M['running_domain_acc'], M['balanced_sample_count']))
                    log('Seg Loss', safe_div(M['running_seg_loss'], M['sample_count']))

            # computing mean_discrepancy
            source_sample, source_seg = random_sample(kMRI(phase, balanced=False, group='source'), configs['MDD_sample_size'])
            target_sample, target_seg = random_sample(kMRI(phase, balanced=False, group='target'), configs['MDD_sample_size'])

            M['epoch_mean_discrepancy'] = model.feature_MDD(source_sample, target_sample)
            M['source_per_class_dice'] = per_class_dice(model(source_sample, seg_only=True), source_seg)
            M['target_per_class_dice'] = per_class_dice(model(target_sample, seg_only=True), target_seg)
            M['epoch_domain_loss'] = safe_div(M['running_domain_loss'], M['balanced_sample_count'])
            M['epoch_domain_acc'] = safe_div(M['running_domain_acc'], M['balanced_sample_count'])
            M['epoch_seg_loss'] = safe_div(M['running_seg_loss'], M['sample_count'])

            pprint(M)
            #update_hyper_param(configs)
            torch.cuda.empty_cache()
            gc.collect()

            with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}-{epoch}.pt'), 'wb+') as f:
                torch.save({
                            'epoch': epoch,
                            'metrics': metrics,
                            'configs': configs,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f)
