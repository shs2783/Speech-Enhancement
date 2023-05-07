'https://arxiv.org/pdf/2206.07293.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_stft import ConvSTFT, ConviSTFT
from modules.complex_nn import ComplexConv2d, ComplexConvTranspose2d, ComplexLSTM, ComplexBatchNorm2d, complex_concat
from modules import CCBAM

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), norm=True, act=True, causal=True, is_complex=True, **kwargs) -> None:
        super().__init__()

        self.causal = causal
        self.padding = padding
        negative_slope = kwargs.get('negative_slope', 0.2)

        if is_complex:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU(negative_slope) if act else nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU(negative_slope) if act else nn.Identity()

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.padding[1], 0, 0, 0))
        else:
            x = F.pad(x, (self.padding[1], self.padding[1], 0, 0))

        return self.act(self.norm(self.conv(x)))

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), norm=True, act=True, causal=True, is_complex=True, **kwargs) -> None:
        super().__init__()

        self.causal = causal
        self.padding = padding
        negative_slope = kwargs.get('negative_slope', 0.2)

        if is_complex:
            self.conv_transposed = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU(negative_slope) if act else nn.Identity()
        else:
            self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU(negative_slope) if act else nn.Identity()

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.padding[1], 0, 0, 0))
        else:
            x = F.pad(x, (self.padding[1], self.padding[1], 0, 0))

        return self.act(self.norm(self.conv_transposed(x)))


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, num_repeats=6, is_complex=True) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [ConvBlock(in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(0, 1), causal=True, is_complex=is_complex)]
            in_channels = out_channels

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs

class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, num_repeats=6, reduction_ratio=16, is_complex=True) -> None:
        super().__init__()

        self.skip_connection_attention_layers = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(num_repeats):
            self.skip_connection_attention_layers += [CCBAM(in_channels, reduction_ratio)]
            self.layers += [ConvTransposeBlock(in_channels * 2, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(0, 0), causal=True, is_complex=is_complex)]
            in_channels = out_channels

    def forward(self, x, encoder_outputs):
        for i, layer in enumerate(self.layers):
            encoder_output = encoder_outputs.pop()
            encoder_output = self.skip_connection_attention_layers[i](encoder_output)

            if x.shape[-1] > encoder_output.shape[-1]:
                x = x[:, :, :, :-1]
            if x.shape[-2] < encoder_output.shape[-2]:
                x = F.pad(x, (0, 0, 0, 1))

            x = complex_concat([x, encoder_output], dim=1)
            x = layer(x)
        return x

class FRCRN(nn.Module):
    def __init__(self, window_size=320, hop_size=160, fft_size=640, lstm_channels=256, reduction_ratio=16, is_complex=True) -> None:
        super().__init__()

        self.stft = ConvSTFT(window_size, hop_size, fft_size)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        self.encoder = Encoder(in_channels=2, out_channels=128, is_complex=is_complex)
        self.decoder = Decoder(in_channels=128, out_channels=128, is_complex=is_complex, reduction_ratio=reduction_ratio)
        self.lstm = ComplexLSTM(256, lstm_channels, num_layers=2, bidirectional=False, batch_first=True)

        self.final_conv = nn.Conv2d(128, 2, kernel_size=(1, 2), bias=False)

        self.fft_size = fft_size

    def forward(self, x):
        ### stft
        complex_specs = self.stft(x)

        noisy_real = complex_specs[:, :self.fft_size // 2 + 1, :]
        noisy_imag = complex_specs[:, self.fft_size // 2 + 1:, :]

        complex_specs = torch.stack([noisy_real, noisy_imag], dim=1)
        complex_specs = complex_specs[:, :, 1:, :]

        ### process
        x = complex_specs
        x, encoder_outputs = self.encoder(x)

        batch, channels, freq, time = x.shape
        x = x.view(batch, channels * freq, time).permute(0, 2, 1)

        x = self.lstm(x)
        x = x.permute(0, 2, 1).view(batch, channels, freq, time)

        x = self.decoder(x, encoder_outputs)
        x = self.final_conv(x)
        x = F.pad(x, (0, 0, 1, 0))

        ### masking
        mask = torch.tanh(x)
        clean_estimate_spec = mask * complex_specs
        clean_estimate_spec = F.pad(clean_estimate_spec, (0, 0, 1, 0))

        ### istft
        real = clean_estimate_spec[:, 0, :, :]
        imag = clean_estimate_spec[:, 1, :, :]

        clean_estimate_spec = torch.cat([real, imag], dim=1)
        clean_estimate_wav = self.istft(clean_estimate_spec)
        clean_estimate_wav = torch.clamp_(clean_estimate_wav, -1, 1)
        return clean_estimate_spec, clean_estimate_wav

if __name__ == '__main__':
    # TODO 1: FSMN, CFSMN

    win_size = 320  # = 16000Hz * 0.02s (20ms) in paper
    hop_size = 160  # = 16000Hz * 0.01s (10ms) in paper
    fft_size = 640

    signal = torch.randn(1, 16000)
    model = FRCRN(win_size, hop_size, fft_size, is_complex=True)

    clean_estimate_spec, clean_estimate_wav = model(signal)
    print(clean_estimate_wav.shape)
