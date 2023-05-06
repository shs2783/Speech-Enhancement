import os
import logging

import numpy as np

import torch
import torchaudio
import torch.nn.functional as F

def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S',
               file=False):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(name) if file else logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def show_params(model):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in model.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size(), end=' ')

                i = params.numel()
                if 'weight' in name:
                    print('>'.rjust(20, '-'), i)
                else:
                    print()
                    
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)

    return num_params

def initialize_params(model, nonlinearity='relu', weight_norm=True):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, torch.nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.nn.init.constant_(module.weight.data, 1)
            torch.nn.init.constant_(module.bias.data, 0)

def train_test_split(len_dataset, train_ratio=0.8, shuffle=True):
    indices = list(range(len_dataset))

    if shuffle:
        np.random.shuffle(indices)

    split = int(train_ratio * len_dataset)
    train_idx, test_idx = indices[:split], indices[split:]

    return train_idx, test_idx

def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()

def reshape_wav_to_mono(wav):
    if wav.dim() == 3:
        batch, channel, length = wav.size()
        wav = wav.view(batch * channel, length)
    return wav

def pad_or_truncate_wav(estimate_wav, target_wav):
    estimate_length = estimate_wav.shape[-1]
    target_length = target_wav.shape[-1]

    if estimate_length < target_length:
        gap = target_length - estimate_length
        estimate_wav = F.pad(estimate_wav, (0, gap))
    elif estimate_length > target_length:
        estimate_wav = estimate_wav[:, :target_length]

    return estimate_wav
