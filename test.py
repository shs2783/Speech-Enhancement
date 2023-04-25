import sys
sys.path.append('models')

import torch
from utils import show_params
from models import DCCRN

if __name__ == '__main__':
    x = torch.randn(1, 16000)
    model = DCCRN('dccrn-CL', window_size=400, hop_size=100, fft_size=512, bidirectional=False, is_complex=True)

    clean_estimate_spec, clean_estimate_wav = model(x)
    print(clean_estimate_wav.shape)
    
    show_params(model)