'https://arxiv.org/pdf/2008.00264.pdf'
'https://github.com/huyanxin/DeepComplexCRN'

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_stft import ConvSTFT, ConviSTFT
from complex_nn import ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, ComplexLSTM, ComplexBatchNorm2d, complex_concat

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), norm=True, act=True, causal=True, is_complex=True, **kwargs) -> None:
        super().__init__()

        self.causal = causal
        self.padding = padding

        if is_complex:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.PReLU() if act else nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.PReLU() if act else nn.Identity()

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

        if is_complex:
            self.conv_transposed = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.PReLU() if act else nn.Identity()
        else:
            self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=(self.padding[0], 0), bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.PReLU() if act else nn.Identity()

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.padding[1], 0, 0, 0))
        else:
            x = F.pad(x, (self.padding[1], self.padding[1], 0, 0))

        return self.act(self.norm(self.conv_transposed(x)))

class LSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, linear_channels, num_layers=2, is_complex=True, **kwargs) -> None:
        super().__init__()

        bidirectional = kwargs.get('bidirectional', True)
        num_direction = 2 if bidirectional else 1

        self.layers = nn.ModuleList()
        if is_complex:
            for _ in range(num_layers-1):
                self.layers += [ComplexLSTM(in_channels, hidden_channels, num_layers=1, **kwargs)]
            self.layers += [ComplexLSTM(num_direction*hidden_channels, hidden_channels, num_layers=1, **kwargs)]
            self.layers += [ComplexLinear(num_direction*hidden_channels, linear_channels)]

        else:
            self.layers += [nn.LSTM(in_channels, hidden_channels, num_layers=num_layers, **kwargs)]
            self.layers += [nn.Linear(num_direction*hidden_channels, linear_channels)]

        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, ComplexLSTM)):
                layer.flatten_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x  # lstm returns (output, (h_n, c_n))

        return x
    
class Encoder(nn.Module):
    def __init__(self, encoder_channels, in_channels=2, is_complex=True) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for out_channels in encoder_channels:
            self.layers += [ConvBlock(in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1), causal=True, is_complex=is_complex)]
            in_channels = out_channels

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs
    
class Decoder(nn.Module):
    def __init__(self, decoder_channels, in_channels=256, is_complex=True) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for out_channels in decoder_channels:
            self.layers += [ConvTransposeBlock(in_channels*2, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0), causal=True, is_complex=is_complex)]
            in_channels = out_channels

    def forward(self, x, encoder_outputs):
        for layer in self.layers:
            encoder_output = encoder_outputs.pop()
            if x.shape[-1] > encoder_output.shape[-1]:
                x = x[:, :, :, :-1]

            x = complex_concat([x, encoder_output], dim=1)
            x = layer(x)
        return x
    
class DCCRN(nn.Module):
    def __init__(self, config='dccrn-CL', window_size=400, hop_size=100, fft_size=512, lstm_channels=256, linear_channels=1024, bidirectional=False, is_complex=True) -> None:
        super().__init__()

        if config == 'dccrn-C' or config == 'dccrn-R' or config == 'dccrn-E':
            encoder_channels = [32, 64, 128, 128, 256, 256]  # DCCRN-R, DCCRN-C, DCCRN-E
            masking = config[-1]
        else:
            encoder_channels = [32, 64, 128, 256, 256, 256]  # DCCRN-CL
            masking = 'E'
        decoder_channels = encoder_channels[:-1][::-1] + [2]

        reduced_freq_dim = fft_size // 2 // (2**(len(encoder_channels)))
        freq_channels = reduced_freq_dim * encoder_channels[-1]

        self.stft = ConvSTFT(window_size, hop_size, fft_size)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        self.encoder = Encoder(encoder_channels, in_channels=2, is_complex=is_complex)
        self.decoder = Decoder(decoder_channels, in_channels=256, is_complex=is_complex)
        self.lstm = LSTMBlock(freq_channels, lstm_channels, linear_channels, num_layers=2, batch_first=True, bidirectional=bidirectional, is_complex=is_complex)

        self.masking = masking
        self.fft_size = fft_size

    def forward(self, x):
        ### stft
        complex_specs = self.stft(x)

        noisy_real = complex_specs[:, :self.fft_size//2 + 1, :]
        noisy_imag = complex_specs[:, self.fft_size//2 + 1:, :]

        complex_specs = torch.stack([noisy_real, noisy_imag], dim=1)
        complex_specs = complex_specs[:, :, 1:, :]

        ### process
        x = complex_specs
        x, encoder_outputs = self.encoder(x)

        batch, channels, freq, time = x.shape
        x = x.view(batch, channels*freq, time).permute(0, 2, 1)

        x = self.lstm(x)
        x = x.permute(0, 2, 1).view(batch, channels, freq, time)

        x = self.decoder(x, encoder_outputs)
        x = F.pad(x, (0, 0, 1, 0))

        ### masking
        mask_real = x[:, 0]
        mask_imag = x[:, 1]

        if mask_real.shape[-1] > noisy_real.shape[-1]:
            mask_real = mask_real[:, :, :-1]
            mask_imag = mask_imag[:, :, :-1]

        real, imag = self._mask_processing(noisy_real, noisy_imag, mask_real, mask_imag)

        ### istft
        clean_estimate_spec = torch.cat([real, imag], dim=1)
        clean_estimate_wav = self.istft(clean_estimate_spec)
        clean_estimate_wav = torch.clamp_(clean_estimate_wav, -1, 1)
        return clean_estimate_spec, clean_estimate_wav

    def _mask_processing(self, noisy_real, noisy_imag, mask_real, mask_imag):
        if self.masking == 'R':
            real = noisy_real * mask_real
            imag = noisy_imag * mask_imag
        elif self.masking == 'C':
            real = noisy_real * mask_real - noisy_imag * mask_imag
            imag = noisy_real * mask_imag + noisy_imag * mask_real
        elif self.masking == 'E':
            noisy_mag, noisy_phase = self._return_mag_phase(noisy_real, noisy_imag)
            mask_mag, mask_phase = self._return_mag_phase(mask_real, mask_imag)

            mask_real = mask_real / mask_mag
            mask_imag = mask_imag / mask_mag

            mask_mag = torch.tanh(mask_mag)
            mask_phase = torch.atan2(mask_imag, mask_real)

            real = noisy_mag * mask_mag * torch.cos(noisy_phase + mask_phase)
            imag = noisy_mag * mask_mag * torch.sin(noisy_phase + mask_phase)

        return real, imag

    def _return_mag_phase(self, real, imag):
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        return mag, phase

if __name__ == '__main__':
    signal = torch.randn(1, 16000)
    model = DCCRN('dccrn-CL', window_size=400, hop_size=100, fft_size=512, bidirectional=False, is_complex=True)

    clean_estimate_spec, clean_estimate_wav = model(signal)
    print(clean_estimate_wav.shape)
