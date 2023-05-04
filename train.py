import sys
sys.path.append('models')

import torch
from torch.utils.data import DistributedSampler

from audio_dataset import TrainAudioDatasets
from audio_dataloader import AudioDataLoader

from models import GCARN
from trainer import Trainer

from hparams import HyperParams
from utils import get_logger, show_params, train_test_split

def main():
    gpu_id = (0, )
    logger = get_logger(__name__)
    hparams = HyperParams()

    ### model
    logger.info('Building the model {}'.format(hparams.model))
    model = GCARN(**hparams.model)
    num_params = show_params(model)

    ### optimizer
    logger.info("Create optimizer {}".format(hparams.optimizer))
    optimizer = torch.optim.Adam(model.parameters(), **hparams.optimizer)

    ### scheduler
    logger.info("Create scheduler {}".format(hparams.scheduler))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **hparams.scheduler)

    ### dataset
    logger.info('Making the train and validation datasets')

    dataset = TrainAudioDatasets(**hparams.dataset)
    train_idx, val_idx = train_test_split(len(dataset), **hparams.split_dataset)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    ### dataloader
    logger.info('Making the train and validation dataloader')

    train_sampler = DistributedSampler(train_dataset) if len(gpu_id) > 1 else None
    test_sampler = None

    train_loader = AudioDataLoader(train_dataset, sampler=train_sampler, **hparams.train_dataloader)
    val_loader = AudioDataLoader(val_dataset, sampler=test_sampler, **hparams.val_dataloader)

    ### trainer
    logger.info('Building the trainer')
    trainer = Trainer(hparams, model, optimizer, scheduler, gpu_id=gpu_id)
    trainer.run(train_loader, val_loader, train_sampler)


if __name__ == "__main__":
    main()
