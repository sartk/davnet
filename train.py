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
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show


def train(**kwargs):

    configs = default_configs.copy()
    configs.update(kwargs)
    N = configs['num_epochs']
    epochs_axis = np.arange(0, N)

    def reload_plots():
        pass

    if configs['plot_progress']:

        seg_loss_train = np.repeat(np.nan, N)
        seg_loss_val = np.repeat(np.nan, N)
        dom_loss_train = np.repeat(np.nan, N)
        dom_loss_val = np.repeat(np.nan, N)
        dom_acc_train = np.repeat(np.nan, N)
        dom_acc_val = np.repeat(np.nan, N)
        plt.ioff()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
        fig.suptitle('Performance')

        def reload_plots():
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                ax.clear()
            ax1.plot(epochs_axis, seg_loss_train, '.-')
            ax2.plot(epochs_axis, seg_loss_val, '.-')
            ax3.plot(epochs_axis, dom_loss_train, '.-')
            ax4.plot(epochs_axis, dom_loss_val, '.-')
            ax5.plot(epochs_axis, dom_acc_train, '.-')
            ax6.plot(epochs_axis, dom_acc_val, '.-')
            plt.draw()

        reload_plots()
        ax1.set_ylabel('Training Segmentation Loss')
        ax2.set_ylabel('Validation Segmentation Loss')
        ax3.set_ylabel('Training Domain Loss')
        ax4.set_ylabel('Validation Domain Loss')
        ax5.set_ylabel('Training Domain Accuracy ')
        ax6.set_ylabel('Training Domain Accuracy')
        ax6.set_xlabel('Epochs')

        plt.show()


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

    best_valid_loss = {'seg': np.inf, 'domain': np.inf}
    patience_counter = 0
    groups = ['balanced']

    for epoch in tracker(range(N), desc='epoch'):

        sample_count = {
            'train': 0,
            'valid': 0
        }
        correct_domain_label = 0
        running_domain_loss = {
            'train': 0,
            'valid': 0,
        }
        running_domain_acc = {
            'train': 0,
            'valid': 0,
        }
        running_seg_loss = {
            'train': 0,
            'valid': 0
        }

        for group in tracker(iter(groups), desc='group'):
            for phase in tracker(iter(configs['phases']), desc='phase'):

                model.train(phase == 'train')
                dataloader = dataloaders[group][phase]
                len_dataloader = len(dataloader)
                i = 0
                for img, seg_label, domain_label in tracker(dataloader, desc='batch'):
                    p = float(i + epoch * len_dataloader) / N / len_dataloader
                    grad_reversal_coef = 2. / (1. + np.exp(-10 * p)) - 1

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
                        seg_pred, domain_pred = model(img, grad_reversal_coef, seg_only=True), torch.tensor([1, 0])

                    seg_loss = F_seg_loss(seg_pred, seg_label)
                    domain_loss = F_domain_loss(domain_pred, domain_label)
                    err = (seg_loss + domain_loss)

                    if phase == 'train':
                        err.backward()
                        optimizer.step()

                    running_domain_acc[phase] += (domain_pred.argmax(-1) == domain_label.argmax(-1)).sum().item()
                    sample_count[phase] += img.size(0)
                    running_seg_loss[phase] += seg_loss.item() * img.size(0)
                    running_domain_loss[phase] += domain_loss.item() * img.size(0)
                    i += 1

        epoch_domain_loss = {}
        epoch_domain_acc = {}
        epoch_seg_loss = {}

        for phase in configs['phases']:
            epoch_domain_loss[phase] = running_domain_loss[phase] / sample_count[phase]
            epoch_domain_acc[phase] = running_domain_acc[phase] / sample_count[phase]
            epoch_seg_loss[phase] = running_seg_loss[phase] / sample_count[phase]
            print('Phase: {}, Epoch: {}, Domain Loss: {:.4f}, Seg Loss: {:.4f}, Domain Acc: {:.4f}'.format(phase, epoch, epoch_domain_loss[phase], epoch_seg_loss[phase], epoch_domain_acc[phase]))

        if epoch == N // 2:
            groups.append('all_source')

        if (epoch_domain_loss['valid'] < best_valid_loss['domain']) and (epoch_seg_loss['valid'] < best_valid_loss['seg']):
            patience_counter = 0
            best_valid_loss['domain'] = epoch_domain_loss['valid']
            epoch_seg_loss['valid'] = best_valid_loss['seg']
            with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}-{phase}-{epoch}.pt'), 'wb+') as f:
                torch.save({
                            'epoch': epoch,
                            'phase': phase,
                            'groups': groups,
                            'domain_loss': epoch_domain_loss,
                            'sample_count': sample_count,
                            'domain_acc': epoch_domain_acc,
                            'seg_loss': epoch_seg_loss,
                            'configs': configs,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f)
        else:
            patience_counter += 1
            if patience_counter < patience:
                print('\nPatience counter {}/{}.'.format(patience_counter, configs['patience']))
            elif patience_counter == configs['patience']:
                print('\nEarly stopping. No improvement after {} Epochs.'.format(patience_counter))
                break

        seg_loss_train[epoch] = epoch_seg_loss['train']
        seg_loss_val[epoch] = epoch_seg_loss['valid']
        dom_loss_train[epoch] = epoch_domain_loss['train']
        dom_loss_val[epoch] = epoch_seg_loss['valid']
        dom_acc_train[epoch] = epoch_domain_acc['train']
        dom_acc_val[epoch] = epoch_domain_acc['valid']

        if configs['plot_progress']:
            reload_plots()
            with open(os.path.join(configs['checkpoint_dir'], f'{timestamp}.pt'), 'wb+') as f:
                fig.savefig(f)

        epoch_domain_loss = None  # reset loss
        epoch_domain_acc = None
        epoch_seg_loss = None
        gc.collect()
