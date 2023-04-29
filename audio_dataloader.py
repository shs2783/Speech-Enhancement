import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from audio_dataset import TrainAudioDatasets

class AudioSpliter:
    '''
       Split the audio. All audio is divided 
       into 4s according to the requirements in the paper.
       input:
             chunk_size: split size
             least: Less than this value will not be read
    '''

    def __init__(self, chunk_size=32000, least_samples=16000, is_train=True):
        super(AudioSpliter, self).__init__()

        self.chunk_size = chunk_size
        self.least_samples = least_samples
        self.is_train = is_train

    def __call__(self, sample):
        return self.split(sample)

    def pad_audio(self, sample, gap):
        return F.pad(sample, (0, gap), mode='constant', value=0)

    def chunk_audio(self, sample, start):
        return sample[:, start: start+self.chunk_size]

    def split(self, sample):
        ''' Split audio sample '''

        sample_length = sample['mix'].shape[-1]
        if sample_length < self.least_samples:
            return []

        if sample_length < self.chunk_size:
            ### pad audio
            gap = self.chunk_size - sample_length

            sample['mix'] = self.pad_audio(sample['mix'], gap)
            sample['ref'] = [self.pad_audio(ref, gap) for ref in sample['ref']]

        else:
            ### chunk audio
            random_start = random.randint(0, sample_length - self.chunk_size) if self.is_train else 0

            sample['mix'] = self.chunk_audio(sample['mix'], random_start)
            sample['ref'] = [self.chunk_audio(ref, random_start) for ref in sample['ref']]

        return [sample]

class AudioDataLoader:
    '''
        Custom dataloader method
        input:
              dataset (Dataset): dataset from which to load the data.
              batch_size (int, optional): how many samples per batch to load
              chunk_size (int, optional): split audio size (default: 32000(4s))
              num_workers (int, optional): how many subprocesses to use for data (default: 4)
              is_train: if this dataloader for training
    '''

    def __init__(self, dataset, batch_size=1, num_workers=4, chunk_size=32000, pin_memory=False, is_train=True):
        super(AudioDataLoader, self).__init__()

        self.dataset = dataset
        self.is_train = is_train

        self.data_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=is_train,
                                      drop_last=is_train,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      collate_fn=self._collate)
        self.spliter = AudioSpliter(chunk_size=chunk_size, least_samples=chunk_size//2, is_train=is_train)

    def _collate(self, batches):
        batch_list = []
        for batch in batches:
            batch_list += self.spliter(batch)

        return batch_list
        # return [self.spliter(batch) for batch in batches]

    def __iter__(self):
        for batch_audio in self.data_loader:
            yield default_collate(batch_audio)

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__": 
    mix_audio_path = 'synth/mix'
    split_audio_path = ['synth/clean', 'synth/noise']
    
    train_dataset = TrainAudioDatasets(mix_audio_path, split_audio_path, sr=16000)
    val_dataset = TrainAudioDatasets(mix_audio_path, split_audio_path, sr=16000)
    
    train_loader = AudioDataLoader(train_dataset, chunk_size=80000, num_workers=0, batch_size=32, is_train=True)
    val_loader = AudioDataLoader(val_dataset, chunk_size=80000, num_workers=0, batch_size=32, is_train=True)

    for n, eg in enumerate(train_loader):
        print(n, eg['mix'].shape, eg['ref'][0].shape)