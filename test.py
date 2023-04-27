import sys
sys.path.append('models')

import torch
from utils import show_params
from models import DCUNet

if __name__ == '__main__':
    window_size = 1024  # 16000Hz * 0.064s (=64ms) in paper
    hop_size = 256  # 16000Hz * 0.016s (=16ms) in paper
    fft_size = 1024  # same as window_size

    model = DCUNet('dcunet20-large', window_size, hop_size, fft_size, is_complex=True)

    x = torch.rand(1, 32000)
    clean_estimate_spec, clean_estimate_wav = model(x)
    print(clean_estimate_wav.shape)

    show_params(model)