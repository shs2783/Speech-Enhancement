import sys
sys.path.append('models')

import torch
from torch.utils.data import DistributedSampler

from audio_dataset import TrainAudioDatasets
from audio_dataloader import AudioDataLoader

from models import GCARN
from trainer import Trainer

from hparams import HyperParams
from utils import get_logger, show_params, split_dataset_index

def main():
    gpu_id = (0, )
    logger = get_logger(__name__)
    hparams = HyperParams()

    ### model
    logger.info('Building the model')
    model = GCARN(window_size=hparams.window_size, hop_size=hparams.hop_size, fft_size=hparams.fft_size, lstm_channels=hparams.lstm_channels)
    num_params = show_params(model)

    ### optimizer
    logger.info("Create optimizer {0}: {1}".format('Adam', hparams.optimizer_kwargs))
    optimizer = torch.optim.Adam(model.parameters(), **hparams.optimizer_kwargs)

    ### scheduler
    logger.info("Create scheduler {0}: {1}".format('ReduceLROnPlateau', hparams.scheduler_kwargs))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **hparams.scheduler_kwargs)

    ### trainer
    logger.info('Building the trainer')
    trainer = Trainer(hparams, model, optimizer, scheduler, gpu_id=gpu_id)

    ### dataset
    logger.info('Making the train and validation dataset')

    dataset = TrainAudioDatasets(hparams.mix_audio_path, hparams.split_audio_path, sr=hparams.sr, k=hparams.k)
    train_idx, val_idx = split_dataset_index(len(dataset), split_ratio=hparams.train_ratio, shuffle=hparams.split_shuffle)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    ### dataloader
    logger.info('Making the train and validation dataloader')

    train_sampler = DistributedSampler(train_dataset) if len(gpu_id) > 1 else None
    train_loader = AudioDataLoader(train_dataset,
                                   chunk_size=hparams.chunk_size,
                                   batch_size=hparams.batch_size,
                                   pin_memory=hparams.pin_memory,
                                   num_workers=hparams.num_workers,
                                   shuffle=hparams.train_shuffle,
                                   drop_last=hparams.train_drop_last,
                                   sampler=train_sampler)

    val_loader = AudioDataLoader(val_dataset,
                                 chunk_size=hparams.chunk_size,
                                 batch_size=hparams.batch_size,
                                 pin_memory=hparams.pin_memory,
                                 num_workers=hparams.num_workers,
                                 shuffle=hparams.test_shuffle,
                                 drop_last=hparams.test_drop_last)

    trainer.run(train_loader, val_loader, train_sampler)


if __name__ == "__main__":
    main()
