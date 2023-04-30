import sys
sys.path.append('models')

import torch
from torch.utils.data import DistributedSampler

from audio_dataset import TrainAudioDatasets
from audio_dataloader import AudioDataLoader

from models import GCARN
from trainer import Trainer

from utils import get_logger
from multiprocessing import cpu_count

def main():
    gpu_id = (0, )
    num_cpu = cpu_count()
    logger = get_logger(__name__)

    ### model
    logger.info('Building the model')
    # model = DCUNet('dcunet20-large')
    model = GCARN(window_size=512, hop_size=256, fft_size=512, lstm_channels=512)

    ### optimizer
    optimizer_kwargs = {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0,
        'amsgrad': False,
    }

    logger.info("Create optimizer {0}: {1}".format('Adam', optimizer_kwargs))
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    ### scheduler
    scheduler_kwargs = {
        'mode': 'min',
        'factor': 0.1,
        'patience': 5,
        'min_lr': 1e-5,
        'verbose': True,
    }

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=h.lr_decay, last_epoch=last_epoch)

    ### trainer
    logger.info('Building the trainer')
    trainer = Trainer(model, optimizer, scheduler, gpu_id=gpu_id, load_checkpoint=None, num_epochs=100)

    ### data loader
    logger.info('Making the train and test data loader')
    train_mix_audio_path = 'synth/mix'
    train_split_audio_path = ['synth/clean', 'synth/noise']

    val_mix_audio_path = 'synth/mix'
    val_split_audio_path = ['synth/clean', 'synth/noise']

    train_dataset = TrainAudioDatasets(train_mix_audio_path, train_split_audio_path, sr=16000)
    val_dataset = TrainAudioDatasets(val_mix_audio_path, val_split_audio_path, sr=16000)

    train_sampler = DistributedSampler(train_dataset) if len(gpu_id) > 1 else None

    train_loader = AudioDataLoader(train_dataset, chunk_size=32000, batch_size=8, pin_memory=True, num_workers=num_cpu//2, shuffle=True, drop_last=True, sampler=train_sampler)
    val_loader = AudioDataLoader(val_dataset, chunk_size=32000, batch_size=8, pin_memory=True, num_workers=num_cpu//2, shuffle=False, drop_last=False)

    trainer.run(train_loader, val_loader, train_sampler)


if __name__ == "__main__":
    main()
