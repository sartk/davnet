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
import pprint
from itertools import cycle
import pdb
from configs import *

def train(**kwargs):

    configs = default_configs.copy()
    configs.update(kwargs)

    N = configs['num_epochs']
    tracker = tqdm if configs['print_progress'] else identity_tracker
    os.environ['CUDA_VISIBLE_DEVICES'] = configs['CUDA_VISIBLE_DEVICES']
    message = configs['message']
    if message:
        message = f'-{message}'
    timestamp = '{}{}'.format(time.strftime("%Y%m%d-%H%M%S"), message)
    log = logger(timestamp)

    log('\nTraining with configs:\n')
    log(pprint.pformat(configs, indent=4))

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

        log('\nEpoch', epoch)

        if epoch == configs['warmup_length']:
            warmup = False

        metrics = {'train': {}, 'valid': {}}

        for phase in metrics:

            log('\nPhase', phase)

            for m in all_metrics:
                metrics[phase][m] = 0

            M = metrics[phase]
            model.train(phase == 'train')
            i = 0
            a = dataloaders['all_source'][phase]
            len_dataloader = len(a)
            b = cycle([(None, None, None)] if warmup else dataloaders['balanced'][phase])
            iterator = tracker(zip(a, b), desc='batch', total=len_dataloader)

            for ((img_a, seg_label, _), (img_b, _, dlab)) in iterator:

                if configs['valid_freq'] and (i + 1) % configs['valid_freq'] == 0:
                    log(f'\nPeriodic Validation on Epoch {epoch}, Iteration {i}')
                    source_dice, target_dice = baseline(100, model)
                    log('Source Valid Dice', source_dice)
                    log('Target Valid Dice', target_dice)

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
                seg_loss = F_seg_loss(seg_pred, seg_label, exp=configs['dice_loss_exp'])

                if warmup:
                    err = seg_loss
                else:
                    try:
                        _, domain_pred = model(img_b, grad_reversal_coef, seg_only=False)
                        domain_loss = F_domain_loss(domain_pred, domain_label)
                        M['running_domain_acc'] += (domain_pred.argmax(1) == domain_label).sum().item()
                        M['balanced_sample_count'] += n
                        M['running_domain_loss'] += (domain_loss * n).item()
                        M['pred_source'] += (domain_pred.argmax(1) == 0).sum().item()
                        M['pred_target'] += (domain_pred.argmax(1) == 1).sum().item()
                        err = seg_loss + domain_loss
                    except ValueError as e:
                        print(e)
                        err = seg_loss

                if phase == 'train':
                    err.backward()
                    optimizer.step()

                M['running_seg_loss'] += (seg_loss * n).item()
                M['sample_count'] += n
                i += 1

                if configs['log_frequency'] and i % configs['log_frequency'] == 0:
                    log(f'\nPeriodic Log on Epoch {epoch}, Iteration {i}')
                    log('Domain Loss',  safe_div(M['running_domain_loss'], M['balanced_sample_count']))
                    log('Domain Acc', safe_div(M['running_domain_acc'], M['balanced_sample_count']))
                    log('Seg Loss', safe_div(M['running_seg_loss'], M['sample_count']))

            M['epoch_domain_loss'] = safe_div(M['running_domain_loss'], M['balanced_sample_count'])
            M['epoch_domain_acc'] = safe_div(M['running_domain_acc'], M['balanced_sample_count'])
            M['epoch_seg_loss'] = safe_div(M['running_seg_loss'], M['sample_count'])

            try:
                log(pprint.pformat(M, indent=4))
            except:
                pprint.pprint(M)
            torch.cuda.empty_cache()
            gc.collect()

        with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}-{epoch}.pt'), 'wb+') as f:
            torch.save({
                        'epoch': epoch,
                        'metrics': metrics,
                        'configs': configs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'architecture': str(model)
                        }, f)
