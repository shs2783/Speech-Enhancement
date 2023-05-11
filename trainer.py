import os
import sys
import gc
import time

from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torchaudio
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from losses import *
from utils import get_logger, reshape_wav_to_mono, pad_or_truncate_wav


class Trainer:
    def __init__(self, hparams, model, optimizer, scheduler, gpu_id=0):
        ### hyper parameters
        self.current_epoch = 0
        self.no_improvement = 0
        self.best_accuracy = 0
        self.best_loss = 1e10

        for attr in hparams.__dir__():
            if not attr.startswith("__"):
                value = getattr(hparams, attr)

                if not callable(value):
                    setattr(self, attr, value)

        ### build the logger object
        self.logger = get_logger(self.checkpoint_dir + "trainer.log", file=False)

        ### GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available.")
        if not isinstance(gpu_id, (tuple, list)):
            gpu_id = (gpu_id, )

        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id[0]))
        self.logger.info("Trainer prepared on {}".format(self.device))

        ### model and optimizer and scheduler
        self.logger.info("Loading model to GPUs: {}".format(gpu_id))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        ### data parallel
        if len(gpu_id) > 1:
            self.model = DistributedDataParallel(self.model, device_ids=gpu_id, output_device=gpu_id[0])
            self.logger.info("Using DistributedDataParallel")

        ### Whether to resume the model
        if self.load_model_path:
            if not os.path.exists(self.load_model_path):
                raise FileNotFoundError("Could not find resume checkpoint: {}".format(self.load_model_path))

            state_dict = torch.load(self.load_model_path, map_location=self.device)
            self.current_epoch = state_dict['epoch']
            self.best_loss = state_dict['best_loss']
            self.best_accuracy = state_dict['best_accuracy']

            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

            self.optimizer.zero_grad()
            self.logger.info(f"Resume from checkpoint {self.load_model_path}")

        ### mkdir save checkpoint
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def predict(self, mix_wav, is_train):
        if is_train:
            estimate_clean_spec, estimate_clean_wav = self.model(mix_wav)
        else:
            with torch.no_grad():
                estimate_clean_spec, estimate_clean_wav = self.model(mix_wav)

        return estimate_clean_spec, estimate_clean_wav

    def learn(self, dataloader, is_train=True):
        start = time.time()
        current_step = 0
        total_loss = 0
        total_accuracy = 0

        train_or_val = 'Training' if is_train else 'Validation'
        pbar = tqdm(dataloader, total=len(dataloader) // dataloader.batch_size)
        for datas in pbar:
            current_step += 1

            # get data
            mix_wav = datas['mix'].to(self.device)
            target_clean_wav = datas['ref'][0].to(self.device)
            target_noise_wav = datas['ref'][1].to(self.device)

            # estimate clean wav
            estimate_clean_spec, estimate_clean_wav = self.predict(mix_wav, is_train)

            # reshape wav
            estimate_clean_wav = reshape_wav_to_mono(estimate_clean_wav)
            target_clean_wav = reshape_wav_to_mono(target_clean_wav)

            # pad or truncate wav
            estimate_clean_wav = pad_or_truncate_wav(estimate_clean_wav, target_clean_wav)

            # calculate loss
            loss = SI_SNR_loss(estimate_clean_wav, target_clean_wav)
            total_loss += loss.item()
            avg_loss = total_loss / current_step
            
            # calculate_accuracy
            accuracy = 0
            total_accuracy += accuracy
            avg_accuracy = total_accuracy / current_step
            
            # back propagation
            self._update_parameters(loss, is_train)

            pbar.set_description('< {} epoch:{:3d}, iter: {:d}, lr: {:.3e}, loss: {:.4f}, accuracy: {:.4f} > '.format(
                train_or_val, self.current_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, avg_accuracy))

            # gc.collect()
            # torch.cuda.empty_cache()

        pbar.close()
        end = time.time()

        self.logger.info('{} epoch {} done, Total time: {:.1f} min'.format(
            train_or_val, self.current_epoch, (end - start) / 60))

        if is_train:
            self.save_wav_file(mix_wav[0].cpu(), 'train_result/mix', current_step)
            self.save_wav_file(target_clean_wav[0].cpu(), 'train_result/ground_clean', current_step)
            self.save_wav_file(target_noise_wav[0].cpu(), 'train_result/ground_noise', current_step)
            self.save_wav_file(estimate_clean_wav[0].detach().cpu(), 'train_result/output_clean', current_step)

        return avg_loss, avg_accuracy

    def train(self, train_dataloader):
        self.logger.info('Training model ......')
        self.model.train()

        train_loss, train_accuracy = self.learn(train_dataloader, is_train=True)
        return train_loss, train_accuracy

    def val(self, val_dataloader):
        self.logger.info('Validation model ......')
        self.model.eval()

        val_loss, val_accuracy = self.learn(val_dataloader, is_train=False)
        return val_loss, val_accuracy

    def run(self, train_dataloader, val_dataloader, train_sampler=None):
        self.logger.info(f"Starting epoch from {self.current_epoch}")

        self.train_loss_list = []
        self.train_accuracy_list = []
        self.val_loss_list = []
        self.val_accuracy_list = []

        while self.current_epoch < self.num_epochs:
            self.current_epoch += 1
            print()

            # set epoch for distributed sampler
            if len(self.gpu_id) > 1:
                train_sampler.set_epoch(self.current_epoch)

            # train and valiadation
            train_loss, train_accuracy = self.train(train_dataloader)
            val_loss, val_accuracy = self.val(val_dataloader)

            # save loss and accuracy
            self.train_loss_list.append(train_loss)
            self.train_accuracy_list.append(train_accuracy)
            self.val_loss_list.append(val_loss)
            self.val_accuracy_list.append(val_accuracy)

            # update best loss and accuracy
            self._update_best_loss(val_loss)
            self._update_best_accuracy(val_accuracy)

            # step scheduler
            self.scheduler.step(val_loss)

            # flush scheduler info
            sys.stdout.flush()

            # save last checkpoint
            self.save_checkpoint(best=False)

            # early stopping
            if self.no_improvement == self.patience:
                self.logger.info(f"Stop training cause no improvement for {self.no_improvement} epochs")
                break

        self.logger.info(f"Training for {self.current_epoch} / {self.num_epochs} epochs done!".format)

        # plot image
        self.plot_image(self.train_loss_list, self.val_loss_list, matrix='loss', save=False)
        self.plot_image(self.train_accuracy_list, self.val_accuracy_list, matrix='accuracy', save=False)

    def _update_parameters(self, loss, is_train):
        if not is_train:
            return

        loss.requires_grad_(True)
        loss.backward()

        if self.clip_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _update_best_loss(self, val_loss):
        if str(val_loss) in ['inf', '-inf', 'nan']:
            import sys
            sys.exit(f'loss is {val_loss}, stop training')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.no_improvement = 0
            self.save_checkpoint(best=True, matrix='loss')
            self.logger.info(f'best loss updated! -> {self.best_loss:.4f}')
        else:
            self.no_improvement += 1

    def _update_best_accuracy(self, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.save_checkpoint(best=True, matrix='accuracy')
            self.logger.info(f'best accuracy updated! -> {self.best_accuracy:.4f}')

    def plot_image(self, train_matrix, val_matrix, matrix='loss', save=False):
        plt.title(f"{matrix} of train and validation")
        plt.plot(train_matrix, 'b-', label=f'train_{matrix}', linewidth=0.8)
        plt.plot(val_matrix, 'c-', label=f'val_{matrix}', linewidth=0.8)

        plt.xlabel('epoch')
        plt.ylabel(matrix)
        plt.legend()

        if save:
            plt.savefig('model_loss.png')
        plt.show()

    def save_wav_file(self, wav, save_dir, current_step=0, sr=16000):
        os.makedirs(save_dir, exist_ok=True)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        save_dir = os.path.join(save_dir, f'epoch{self.current_epoch}_step{current_step}.wav')
        torchaudio.save(save_dir, wav, sr)

    def save_checkpoint(self, best=True, matrix=''):
        best_or_last = 'best' if best else 'last'
        matrix = f'_{matrix}' if matrix else ''
        file_name = f'{best_or_last}_model{matrix}.pt'

        state_dict = {
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        torch.save(
            state_dict,
            self.checkpoint_dir / file_name
        )
