import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from audio_dataset import TrainAudioDatasets

class AudioSpliter:
    '''
      chunk_size (int): split audio size (default: 32000)
      least_samples (int): Less than this value will not be read (default: 16000)
    '''

    def __init__(self, chunk_size=32000, least_samples=16000):
        super(AudioSpliter, self).__init__()

        self.chunk_size = chunk_size
        self.least_samples = least_samples

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
            random_start = random.randint(0, sample_length - self.chunk_size)

            sample['mix'] = self.chunk_audio(sample['mix'], random_start)
            sample['ref'] = [self.chunk_audio(ref, random_start) for ref in sample['ref']]

        return [sample]

class AudioDataLoader:
    '''
      dataset (Dataset): dataset from which to load the data.
      chunk_size (int): split audio size (default: 32000)
      least_samples (int): Less than this value will not be read (default: 16000)
    '''

    def __init__(self, dataset, chunk_size=32000, least_samples=16000, **kwargs):
        super(AudioDataLoader, self).__init__()

        self.dataset = dataset
        self.batch_size = kwargs['batch_size']

        self.data_loader = DataLoader(dataset, collate_fn=self._collate, **kwargs)
        self.spliter = AudioSpliter(chunk_size, least_samples)

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
    
    train_loader = AudioDataLoader(train_dataset, chunk_size=80000, num_workers=0, batch_size=32)
    val_loader = AudioDataLoader(val_dataset, chunk_size=80000, num_workers=0, batch_size=32)

    for n, eg in enumerate(train_loader):
        print(n, eg['mix'].shape, eg['ref'][0].shape)