from multiprocessing import cpu_count
from hparams.optimizer_hparams import OptimizerHparams
from hparams.scheduler_hparams import SchedulerHparams

class HyperParams:
    # training
    batch_size = 16
    num_epochs = 100
    lr = 1e-3
    patience = 5
    current_epoch = 0
    no_improvement = 0
    best_loss = 1e10
    clip_norm = 0.5
    seed = 2023

    # checkpoint
    checkpoint_dir = 'checkpoint/'
    load_model_path = None

    # model
    window_size = 512
    hop_size = 256
    fft_size = 512
    lstm_channels = 512

    # optimizer
    optimizer_kwargs = OptimizerHparams.adam(lr=lr)

    # scheduler
    scheduler_kwargs = SchedulerHparams.reduce_lr_on_plateau(patience=patience, min_lr=1e-5)

    # dataset
    mix_audio_path = 'synth/mix'
    split_audio_path = ['synth/clean', 'synth/noise']
    k = 5  # only use when mix_audio_path is None
    sr = 16000
    train_ratio = 0.8
    split_shuffle = True

    # dataloader
    chunk_size = 32000
    pin_memory = True,
    num_workers = cpu_count() // 2
    train_shuffle = True
    train_drop_last = True
    test_shuffle = False
    test_drop_last = False