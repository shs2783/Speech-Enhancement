import sys
sys.path.append('models')

import torch
from utils import show_params
from models import CARN, GCARN

if __name__ == '__main__':
    signal = torch.randn(1, 16000)
    # model = CARN(window_size=320, hop_size=160, fft_size=512, lstm_channels=512)
    model = GCARN(window_size=320, hop_size=160, fft_size=512, lstm_channels=512)

    clean_estimate_spec, clean_estimate_wav = model(signal)
    print(clean_estimate_wav.shape)
    
    show_params(model)