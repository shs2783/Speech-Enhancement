import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window

def init_kernels(window_size, fft_size, win_type='hann', inverse=False):
    N = fft_size
    
    window = get_window(win_type, window_size, fftbins=True)
    fourier_basis = np.fft.rfft(np.eye(N))[:window_size]

    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], axis=1).T

    if inverse:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    window = window[None, :, None]

    kernel = torch.from_numpy(kernel.astype(np.float32))
    window = torch.from_numpy(window.astype(np.float32))
    return kernel, window


class ConvSTFT(nn.Module):
    def __init__(self, window_size, hop_size, fft_size=None, win_type='hann', center=True, return_mag_phase=False, fix=True):
        super(ConvSTFT, self).__init__()

        if fft_size is None:
            self.fft_size = window_size
        else:
            self.fft_size = fft_size

        kernel, _ = init_kernels(window_size, self.fft_size, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)

        self.hop_size = hop_size
        self.window_size = window_size
        self.center = center
        self.return_mag_phase = return_mag_phase
        self.pad = self.fft_size // 2

    def forward(self, inputs):
        if inputs.dim() == 1:
            inputs = inputs[None, None, :]
        elif inputs.dim() == 2:
            inputs = inputs[:, None, :]

        if self.center:
            inputs = F.pad(inputs, [self.pad, self.pad], mode='reflect')
        outputs = F.conv1d(inputs, self.weight, stride=self.hop_size)

        if self.return_mag_phase:
            dim = self.fft_size // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase
        else:
            return outputs


class ConviSTFT(nn.Module):
    def __init__(self, window_size, hop_size, fft_size=None, win_type='hann', center=True, fix=True):
        super(ConviSTFT, self).__init__()

        if fft_size is None:
            self.fft_size = window_size
        else:
            self.fft_size = fft_size

        kernel, window = init_kernels(window_size, self.fft_size, win_type, inverse=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(window_size)[:, None, :])

        self.hop_size = hop_size
        self.window_size = window_size
        self.center = center
        self.pad = self.fft_size // 2

    def forward(self, inputs, phase=None, output_length=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        output_length: (int) adjust the length of output
        """

        if phase is not None:
            mags = inputs
            real = mags * torch.cos(phase)
            imag = mags * torch.sin(phase)
            inputs = torch.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.hop_size)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.hop_size)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)

        if self.center:
            outputs = outputs[..., self.pad:]
            outputs = outputs[..., :-self.pad] if output_length is None else outputs

        if output_length is not None:
            outputs = outputs[..., :output_length]

        return outputs.squeeze(1)