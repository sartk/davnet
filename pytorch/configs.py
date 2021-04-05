default_configs = {
    'balanced_batch_size': 8,
    'all_source_batch_size': 16,
    'learning_rate':  1e-4,
    'seg_loss': 'weighted_dice',
    'domain_loss': 'nll',
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
    'log_frequency': 100,
    'MDD_sample_size': 10,
    'domain_loss_weight': 1,
    'disc_in': [3, 4, 5, 6],
    'valid_freq': 500,
    'message': '',
    'dice_weights': None,
    'dice_loss_exp': 0.7,
    'batchnorm': False
}