'https://engineering.linecorp.com/ko/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python'
'https://github.com/Sato-Kunihiko/audio-SNR'

import os
import torch
import random
import numpy as np
import torchaudio

from typing import Optional, Union
from pathlib import Path
from utils import inspect_file

def get_rms(signal: torch.Tensor):
    return torch.sqrt(torch.mean(signal ** 2, axis=-1, keepdim=True))

def get_adjusted_rms(clean_rms: Union[int, float], snr: Union[int, float]):
    return clean_rms / 10 ** (snr/20)

def get_noisy_data(clean_audio_path: Union[str, torch.Tensor],
                   noise_audio_path: Union[str, torch.Tensor],
                   mix_output_path: str = 'synth/mix/', 
                   clean_output_path: str = 'synth/clean/', 
                   noise_output_path: str = 'synth/noise/',
                   mix_sample_rate: int = 16000,
                   clean_sample_rate: Optional[int] = None,
                   noise_sample_rate: Optional[int] = None,
                   noise_repeat: Optional[int] = None,
                   k: int = 100,
                   save: bool = False):
    '''
    clean_audio_path: str or torch tensor shape (channels, samples)
    noise_audio_path: str or torch tensor shape (channels, samples)
    '''

    if save:
        os.makedirs(mix_output_path, exist_ok=True)
        os.makedirs(clean_output_path, exist_ok=True)
        os.makedirs(noise_output_path, exist_ok=True)
        
        mix_output_path = Path(mix_output_path)
        clean_output_path = Path(clean_output_path)
        noise_output_path = Path(noise_output_path)
        
    if isinstance(clean_audio_path, (str, Path)):
        if isinstance(clean_audio_path, str):
            clean_audio_path = Path(clean_audio_path)
            noise_audio_path = Path(noise_audio_path)

        # inspect_file(clean_audio_path)
        # inspect_file(noise_audio_path)

        clean_amp, clean_sample_rate = torchaudio.load(clean_audio_path)
        noise_amp, noise_sample_rate = torchaudio.load(noise_audio_path)

    else:  # in case instance of torch.Tensor
        clean_amp = clean_audio_path
        noise_amp = noise_audio_path

        if clean_amp.ndim == 1:
            clean_amp = clean_amp.unsqueeze(0)
        if noise_amp.ndim == 1:
            noise_amp = noise_amp.unsqueeze(0)

    ### Support only mono audio
    if clean_amp.shape[0] > 1:
        clean_amp = clean_amp.mean(axis=0, keepdim=True)
    if noise_amp.shape[0] > 1:
        noise_amp = noise_amp.mean(axis=0, keepdim=True)

    ### Resample audio
    if clean_sample_rate != mix_sample_rate:
        resampler = torchaudio.transforms.Resample(clean_sample_rate, mix_sample_rate)
        clean_amp = resampler(clean_amp)
    if noise_sample_rate != mix_sample_rate:
        resampler = torchaudio.transforms.Resample(noise_sample_rate, mix_sample_rate)
        noise_amp = resampler(noise_amp)

    outputs = {
        'mixed_output': [],
        'adjusted_noise': [],
        'repeat_noise': [],
        'noise_indices': [],
        'snr': [],
    }

    for i in range(k):
        if noise_amp.shape[1] > clean_amp.shape[1]:
            start = random.randint(0, noise_amp.shape[1] - clean_amp.shape[1])
            end = start + clean_amp.shape[1]
            split_noise_amp = noise_amp[:, start:end]
        else:
            split_noise_amp = noise_amp[:]

        clean_rms = get_rms(clean_amp)
        noise_rms = get_rms(split_noise_amp)

        snr = random.randint(-20, 20)  # random snr db
        adjusted_noise_rms = get_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)

        ### Repeat noise
        repeat_noise_amp = torch.zeros_like(clean_amp)
        clean_length = clean_amp.shape[1]
        noise_length = adjusted_noise_amp.shape[1]
        max_repeat = clean_length // noise_length

        if noise_repeat is not None:
            noise_indices = []
            noise_repeat_num = min(noise_repeat, max_repeat)
            for _ in range(noise_repeat_num):
                start = random.randint(0, clean_length - noise_length)
                end = start + noise_length
                noise_indices.append([start, end])
                repeat_noise_amp[:, start:end] += adjusted_noise_amp
        else:
            noise_repeat_num = max_repeat
            start = 0
            end = noise_repeat_num*noise_length
            noise_indices = [[i, i + noise_length] for i in range(0, end, noise_length)]
            repeat_noise_amp[:, start:end] += adjusted_noise_amp.repeat((1, noise_repeat_num))

        mixed_amp = clean_amp + repeat_noise_amp

        ### Avoid clipping noise
        # max_int16 = np.iinfo(np.int16).max
        # min_int16 = np.iinfo(np.int16).min
        # if mixed_amp.max(dim=1)[0] > max_int16 or mixed_amp.min(dim=1)[0] < min_int16:
        #     if mixed_amp.max(dim=1)[0] >= abs(mixed_amp.min(dim=1)[0]):
        #         reduction_rate = max_int16 / mixed_amp.max(dim=1)
        #     else:
        #         reduction_rate = min_int16 / mixed_amp.min(dim=1)
        #     mixed_amp = mixed_amp * (reduction_rate)
        #     clean_amp = clean_amp * (reduction_rate)

        outputs['mixed_output'].append(mixed_amp)
        outputs['adjusted_noise'].append(noise_amp)
        outputs['repeat_noise'].append(repeat_noise_amp)
        outputs['noise_indices'].append(noise_indices)
        outputs['snr'].append(snr)

        if save:
            file_name = clean_audio_path.stem + '_' + noise_audio_path.stem
            torchaudio.save(mix_output_path / (file_name + f'_mix_{str(i)}.wav'), mixed_amp, mix_sample_rate, encoding="PCM_S", bits_per_sample=16)
            torchaudio.save(clean_output_path / (file_name + f'_clean_{str(i)}.wav'), clean_amp, mix_sample_rate, encoding="PCM_S", bits_per_sample=16)
            torchaudio.save(noise_output_path / (file_name + f'_noise_{str(i)}.wav'), repeat_noise_amp, mix_sample_rate, encoding="PCM_S", bits_per_sample=16)
            # inspect_file(mix_output_path/ (file_name + f'_mix_{str(i)}.wav'))

    return clean_amp, noise_amp, outputs


if __name__ == '__main__':
    clean_amp, noise_amp, outputs = get_noisy_data(clean_audio_path='dataset/LJ001-0034.wav',
                                                   noise_audio_path='dataset/46656-6-5-0.wav',
                                                   mix_output_path='./inference/mix/',
                                                   clean_output_path='./temp/',
                                                   noise_output_path='./temp/',
                                                   noise_repeat=None, k=10, save=True)

    for i in range(len(outputs['mixed_output'])):
        mixed_output = outputs['mixed_output'][i]
        adjusted_noise = outputs['adjusted_noise'][i]
        repeat_noise = outputs['repeat_noise'][i]
        noise_indices = outputs['noise_indices'][i]
        snr = outputs['snr'][i]

        print(mixed_output.shape, adjusted_noise.shape, repeat_noise.shape, noise_indices, snr)