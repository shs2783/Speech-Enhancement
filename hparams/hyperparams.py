from multiprocessing import cpu_count
from hparams.optimizer_hparams import OptimizerHparams
from hparams.scheduler_hparams import SchedulerHparams

class HyperParams:
    # training
    batch_size = 8
    num_epochs = 100
    lr = 1e-3
    clip_norm = 0.5
    patience = 10
    seed = 2023

    # checkpoint
    checkpoint_dir = 'checkpoint/'
    load_model_path = None

    # model
    model = {
        'window_size': 320,
        'hop_size': 160,
        'fft_size': 640,
        'lstm_channels': 256,
    }

    # optimizer
    optimizer = {
        'lr': lr,
        'weight_decay': 1e-2,
    }

    # scheduler
    scheduler = {
        'patience': 5,
        'min_lr': 1e-5,
    }

    optimizer = OptimizerHparams.adamw(**optimizer)
    scheduler = SchedulerHparams.reduce_lr_on_plateau(**scheduler)

    # dataset
    dataset = {
        'mix_audio_path': 'synth/mix',
        'split_audio_path': ['synth/clean', 'synth/noise'],
        'k': 5,  # only use when mix_audio_path is None
        'sr': 16000,
    }

    split_dataset = {
        'train_ratio': 0.8,
        'shuffle': True,
    }

    # dataloader
    train_dataloader = {
        'chunk_size': 32000,
        'batch_size': batch_size,
        'num_workers': cpu_count() // 2,
        'pin_memory': True,
        'shuffle': True,
        'drop_last': True,
    }

    val_dataloader = {
        'chunk_size': 32000,
        'batch_size': batch_size,
        'num_workers': cpu_count() // 2,
        'pin_memory': True,
        'shuffle': False,
        'drop_last': False,
    }