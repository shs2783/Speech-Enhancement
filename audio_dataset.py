import os
from tqdm import tqdm
from pathlib import Path

import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from typing import Union, Optional

from mix_audio import get_noisy_data

def resample_audio(audio, sr, target_sr):
    resampler = transforms.Resample(sr, target_sr)
    return resampler(audio), target_sr

def load_audio(audio_path, target_sr=None):
    audio, sr = torchaudio.load(audio_path)
    if target_sr is not None and sr != target_sr:
        audio, sr = resample_audio(audio, sr, target_sr)
        
    return audio

class TrainAudioDatasets(Dataset):
    def __init__(self,
                 mix_audio_path: Optional[str] = None,
                 split_audio_path: Union[list, str] = ['dataset/clean', 'dataset/noise'],
                 sr: int = 16000,
                 k: int = 10,
                 ):

        self.sr = sr

        ### generate mix audio
        if mix_audio_path is None:
            assert len(split_audio_path) == 2, 'length of split audio path must be 2'
            print('no mix audio data, generating mix audio...')

            sp_list1 = os.listdir(split_audio_path[0])
            sp_list2 = os.listdir(split_audio_path[1])

            pbar = tqdm(sp_list1)
            for sp1 in pbar:
                for sp2 in sp_list2:
                    clean_path = os.path.join(split_audio_path[0], sp1)
                    noise_path = os.path.join(split_audio_path[1], sp2)
                    
                    pbar.set_description(f'clean: {clean_path} | noise: {noise_path}')
                    get_noisy_data(clean_path, noise_path, mix_sample_rate=sr, k=k, save=True)
            pbar.close()
            
            mix_audio_path = 'synth/mix'
            split_audio_path = ['synth/clean', 'synth/noise']

        if isinstance(split_audio_path, str):
            split_audio_path = [split_audio_path]

        ### mix audio list
        self.mix_audio_path = Path(mix_audio_path)
        self.mix_audio_list = os.listdir(mix_audio_path)

        ### split audio list
        self.split_audio_path = [Path(audio_path) for audio_path in split_audio_path]
        self.split_audio_list = [os.listdir(audio_path) for audio_path in split_audio_path]

    def __len__(self):
        return len(self.mix_audio_list)

    def __getitem__(self, idx):
        mix_audio = load_audio(self.mix_audio_path / self.mix_audio_list[idx], self.sr)

        split_audio = [
            load_audio(self.split_audio_path[i] / self.split_audio_list[i][idx], self.sr)
            for i in range(len(self.split_audio_path))
        ]

        return {
            'mix': mix_audio,
            'ref': split_audio
        }

class TestAudioDatasets(Dataset):
    def __init__(self, 
                 mix_audio_path: str,
                 sr: int = 16000
                 ):
        
        self.sr = sr
        self.mix_audio_path = Path(mix_audio_path)
        
        if self.mix_audio_path.is_file():
            self.mix_audio_list = [self.mix_audio_path.name]
            self.mix_audio_path = self.mix_audio_path.parent
        elif self.mix_audio_path.is_dir():
            self.mix_audio_list = os.listdir(mix_audio_path)
        else:
            raise ValueError('mix audio path is not a file or directory')
        
    def __len__(self):
        return len(self.mix_audio_list)

    def __getitem__(self, idx):
        file_name = self.mix_audio_list[idx][:-4]
        mix_audio = load_audio(self.mix_audio_path / self.mix_audio_list[idx], self.sr)
        return file_name, mix_audio

if __name__ == '__main__':
    mix_audio_path = None
    split_audio_path = ['dataset/split/clean', 'dataset/split/noise']
    # mix_audio_path = 'synth/mix'
    # split_audio_path = ['synth/clean', 'synth/noise']

    audio_dataset = TrainAudioDatasets(mix_audio_path, split_audio_path, k=5, sr=16000)
    print(len(audio_dataset))
    
    for samples in audio_dataset:
        print(samples['mix'].shape, samples['ref'][0].shape, samples['ref'][1].shape)