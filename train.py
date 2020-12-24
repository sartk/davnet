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
from evaluate import evaluate

def train(**kwargs):

    configs = default_configs.copy()
    configs.update(**kwargs)
    tracker = tqdm if configs['print_progress'] else identity_tracker
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    dataloaders = {
        'balanced': {
            'train': DataLoader(dataset=kMRI('train', balanced=True), batch_size=configs['balanced_batch_size'], shuffle=True, num_workers=configs['num_workers']),
            'valid': DataLoader(dataset=kMRI('valid', balanced=True), batch_size=configs['balanced_batch_size'], shuffle=True, num_workers=configs['num_workers'])
        },
        'source_only': {
            'train': DataLoader(dataset=kMRI('train', balanced=False, group='source_only'), batch_size=2 * configs['source_only_batch_size'], shuffle=True, num_workers=configs['num_workers']),
            'valid': DataLoader(dataset=kMRI('valid', balanced=False, group='source_only'), batch_size=2 * configs['source_only_batch_size'], shuffle=True, num_workers=configs['num_workers'])
        }
    }

    model = configs['model'](classes=configs['classes'])
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

    #best_valid_loss = np.inf
    #best_valid_acc = 0
    #patience_counter = 0

    for epoch in tracker(range(configs['num_epochs']), desc='epoch'):

        sample_count = 0
        domain_running_loss = 0
        seg_running_loss = 0

        for group in tracker(iter(['balanced', 'all_source']), desc='group'):
            for phase in tracker(iter(['train', 'valid']), desc='phase'):

                model.train(phase == 'train')
                dataloader = tracker(dataloaders[group][phase], desc='batch')

                for i, (img, seg_label, domain_label) in enumerate(dataloader):

                    p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
                    grad_reversal_coef = 2. / (1. + np.exp(-10 * p)) - 1

                    data_type = torch.HalfTensor if configs['half_precision'] else torch.FloatTensor
                    img = img.type(data_type)

                    if configs['cuda']:
                        img = img.cuda(non_blocking=True)
                        seg_label = seg_label.cuda(non_blocking=True)
                        domain_label = domain_label.cuda(non_blocking=True)

                    optimizer.zero_grad()

                    if group == 'balanced':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=False)
                    elif group == 'all_source':
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=True), torch.tensor([1, 0])

                    seg_loss = F_seg_loss(seg_pred, seg_label)
                    domain_loss = F_domain_loss(domain_pred, domain_label)
                    err = seg_loss + domain_loss

                    if phase == 'train':
                        err.backward()
                        optimizer.step()

                    sample_count += img.size(0)
                    seg_running_loss += seg_loss.item() * img.size(0)
                    domain_running_loss += domain_loss.item() * img.size(0)

        epoch_domain_loss = running_domain_loss / sample_count
        epoch_seg_loss = running_seg_loss / sample_count

        if configs['print_progress']:
            print('{} Epoch: {}, Domain Loss: {:.4f}, Seg Loss: {:.4f}'.format(phase, epoch, epoch_domain_loss, epoch_seg_loss))

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'domain_loss': epoch_domain_loss,
                    'seg_loss': epoch_seg_loss
                    }, os.path.join(configs['checkpoint_dir'], f'{timestamp}-{epoch}.pt'))

        gc.collect()
