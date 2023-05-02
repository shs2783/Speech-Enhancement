import os
import sys
import gc
import time

from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torchaudio
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from losses import *
from utils import get_logger, show_params


class Trainer:
    def __init__(self, hparams, model, optimizer, scheduler, gpu_id=0):

        ### hyper parameters
        for attr in hparams.__dir__():
            if not attr.startswith("__"):
                value = getattr(hparams, attr)
        
                if not callable(value):
                    setattr(self, attr, value)
        
        ### build the logger object
        self.logger = get_logger(self.checkpoint_dir + "trainer.log", file=False)

        ### GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable... exist")
        if not isinstance(gpu_id, (tuple, list)):
            gpu_id = (gpu_id, )

        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id[0]))
        self.logger.info("Trainer prepared on {}".format(self.device))

        ### model and optimizer and scheduler
        self.logger.info("Loading model to GPUs:{}".format(gpu_id))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        ### data parallel
        if len(gpu_id) > 1:
            self.model = DistributedDataParallel(self.model, device_ids=gpu_id, output_device=gpu_id[0])
            self.logger.info("Using DistributedDataParallel")

        ### check model parameters
        num_params = show_params(self.model)

        ### Whether to resume the model
        if self.load_model_path:
            if not os.path.exists(self.load_model_path):
                raise FileNotFoundError("Could not find resume checkpoint: {}".format(self.load_model_path))

            state_dict = torch.load(self.load_model_path, map_location=self.device)
            self.best_loss = state_dict['best_loss']
            self.current_epoch = state_dict['epoch'] + 1

            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.logger.info(f"Resume from checkpoint {self.load_model_path}: epoch {self.current_epoch}")

        ### mkdir save checkpoint
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def learn(self, dataloader, is_val=False):
        start = time.time()
        current_step = 0
        total_loss = 0

        pbar = tqdm(dataloader, total=len(dataloader) // dataloader.batch_size)
        for datas in pbar:
            current_step += 1

            # get data
            mix_wav = datas['mix'].to(self.device)
            target_clean_wav = datas['ref'][0].to(self.device)
            target_noise_wav = datas['ref'][1].to(self.device)

            # estimate clean wav
            if is_val:
                with torch.no_grad():
                    estimate_clean_spec, estimate_clean_wav = self.model(mix_wav)
            else:
                estimate_clean_spec, estimate_clean_wav = self.model(mix_wav)

            # reshape to 2D
            if estimate_clean_wav.dim() == 3:
                batch, channel, length = estimate_clean_wav.size()
                estimate_clean_wav = estimate_clean_wav.view(batch*channel, length)
            if target_clean_wav.dim() == 3:
                batch, channel, length = target_clean_wav.size()
                target_clean_wav = target_clean_wav.view(batch*channel, length)

            # pad or truncate
            estimate_clean_wav = self.pad_or_truncate_wav(estimate_clean_wav, target_clean_wav)

            # calculate loss
            self.optimizer.zero_grad()
            loss = SI_SNR_loss(estimate_clean_wav, target_clean_wav)
            loss.requires_grad_(True)
            loss.backward()

            # update parameters
            if self.clip_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # total loss and avg loss
            total_loss += loss.item()
            avg_loss = total_loss / current_step

            pbar.set_description('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}> '.format(
                self.current_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss))

            # gc.collect()
            # torch.cuda.empty_cache()

        pbar.close()
        end = time.time()
        total_avg_loss = total_loss / current_step
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.current_epoch, self.optimizer.param_groups[0]['lr'], total_avg_loss, (end - start) / 60))

        if not is_val:
            self.save_wav_file(mix_wav[0].cpu(), 'train_result/mix', current_step)
            self.save_wav_file(target_clean_wav[0].cpu(), 'train_result/ground_clean', current_step)
            self.save_wav_file(target_noise_wav[0].cpu(), 'train_result/ground_noise', current_step)
            self.save_wav_file(estimate_clean_wav[0].detach().cpu(), 'train_result/output_clean', current_step)

        return total_avg_loss

    def train(self, train_dataloader):
        self.logger.info('Training model ......')
        self.model.train()

        train_loss = self.learn(train_dataloader)
        return train_loss

    def val(self, val_dataloader):
        self.logger.info('Validation model ......')
        self.model.eval()

        val_loss = self.learn(val_dataloader, is_val=True)
        return val_loss

    def run(self, train_dataloader, val_dataloader, train_sampler=None):
        self.logger.info(f"Starting epoch from {self.current_epoch}")

        train_losses = []
        val_losses = []

        while self.current_epoch < self.num_epochs:
            self.current_epoch += 1

            # set epoch for distributed sampler
            if len(self.gpu_id) > 1:
                train_sampler.set_epoch(self.current_epoch)

            # train and valiadation
            train_loss = self.train(train_dataloader)
            val_loss = self.val(val_dataloader)

            # save loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # update best loss
            self._update_best_loss(val_loss)

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

        self.logger.info(f"Training for {self.current_epoch} /{self.num_epochs} epoches done!".format)

        # loss image
        self.plot_loss_image(train_losses, val_losses, save=False)

    def _update_best_loss(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.no_improvement = 0
            self.save_checkpoint(best=True)
            self.logger.info(f'Epoch: {self.current_epoch}, now best loss change: {self.best_loss:.4f}')
        else:
            self.no_improvement += 1
            self.logger.info(f'no improvement, best loss: {self.scheduler.best:.4f}')

    def pad_or_truncate_wav(self, estimate_wav, target_wav):
        estimate_length = estimate_wav.shape[-1]
        target_length = target_wav.shape[-1]

        if estimate_length < target_length:
            gap = target_length - estimate_length
            estimate_wav = F.pad(estimate_wav, (0, gap))
        elif estimate_length > target_length:
            estimate_wav = estimate_wav[:, :target_length]

        return estimate_wav

    def save_wav_file(self, wav, save_dir, current_step=0, sr=16000):
        os.makedirs(save_dir, exist_ok=True)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        save_dir = os.path.join(save_dir, f'epoch{self.current_epoch}_step{current_step}.wav')
        torchaudio.save(save_dir, wav, sr)

    def plot_loss_image(self, train_losses, val_losses, save=False):
        plt.title("Loss of train and test")
        plt.plot(train_losses, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(val_losses, 'c-', label=u'val_loss', linewidth=0.8)

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        if save:
            plt.savefig('model_loss.png')
        plt.show()

    def save_checkpoint(self, best=True):
        state_dict = {
            "best_loss": self.best_loss,
            "epoch": self.current_epoch,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        torch.save(
            state_dict,
            self.checkpoint_dir / '{}.pt'.format("best_model" if best else "last_model")
        )
