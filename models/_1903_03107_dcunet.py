'https://openreview.net/pdf?id=SkeRTsAcYm'
'https://github.com/pheepa/DCUnet/blob/master/dcunet.ipynb'

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_stft import ConvSTFT, ConviSTFT
from modules.complex_nn import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexLeakyReLU
from architectures import dcunet_architecture

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, is_complex=True, **kwargs) -> None:
        super().__init__()
        
        negative_slope = kwargs.get('negative_slope', 0.01)
        if is_complex:
            self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = ComplexLeakyReLU(negative_slope) if act else nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU(negative_slope) if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, is_complex=True, **kwargs) -> None:
        super().__init__()

        if is_complex:
            self.conv_transposed = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
            self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = ComplexLeakyReLU() if act else nn.Identity()
        else:
            self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
            self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
            self.act = nn.LeakyReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv_transposed(x)))

class Encoder(nn.Module):
    def __init__(self, architecture, in_channels=32, is_complex=True) -> None:
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(len(architecture)):
            out_channels, kernel_size, stride, padding = architecture[i]
            out_channels = out_channels[0] * 2 if is_complex else out_channels[1]

            layers += [ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, is_complex=is_complex)]
            in_channels = out_channels

        self.layers = layers

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return x, outputs

class Decoder(nn.Module):
    def __init__(self, architecture, in_channels=64, mask_channels=2, is_complex=True) -> None:
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(len(architecture)-1):
            kernel_size, stride, padding = architecture[-i-1][1:]
            out_channels = architecture[-i-2][0]
            out_channels = out_channels[0] * 2 if is_complex else out_channels[1]

            layers += [ConvTransposeBlock(in_channels*2, out_channels, kernel_size, stride=stride, padding=padding, is_complex=is_complex)]
            in_channels = out_channels

        kernel_size, stride, padding = architecture[0][1:]
        layers += [ConvTransposeBlock(in_channels*2, mask_channels, kernel_size, stride=stride, padding=padding, is_complex=is_complex, act=False)]
        self.layers = layers

    def forward(self, x, encoder_outputs=None):
        for layer in self.layers:
            if encoder_outputs is not None:
                encoder_output = encoder_outputs.pop()
                
                if encoder_output.shape != x.shape:
                    h_diff = abs(encoder_output.shape[2] - x.shape[2])
                    w_diff = abs(encoder_output.shape[3] - x.shape[3])
                    x = F.pad(x, (0, w_diff, 0, h_diff))
                x = torch.cat([x, encoder_output], dim=1)
                
            x = layer(x)
        return x
    
class DCUNet(nn.Module):
    def __init__(self, config, window_size=512, hop_size=128, fft_size=512, normalize=False, is_complex=True) -> None:
        super().__init__()
        
        architecture = dcunet_architecture[config]
        enc_channels = architecture[0][0][0] * 2 if is_complex else architecture[0][0][1]
        dec_channels = architecture[-1][0][0] * 2 if is_complex else architecture[-1][0][1]
        mask_channels = 2 if is_complex else 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.normalize = normalize

        self.stft = ConvSTFT(window_size, hop_size, fft_size)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        self.first_conv = ConvBlock(in_channels=mask_channels, out_channels=enc_channels, kernel_size=3, padding=1, is_complex=is_complex)
        self.encoder = Encoder(architecture, enc_channels, is_complex)
        self.decoder = Decoder(architecture, dec_channels, mask_channels, is_complex)
        
    def forward(self, x):
        ### stft
        complex_specs = self.stft(x)  # (batch, freq, time)

        real = complex_specs[:, :self.fft_size//2 + 1]
        imag = complex_specs[:, self.fft_size//2 + 1:]
        complex_specs = torch.stack([real, imag], dim=1)  # (batch, 2, freq, time)  ## 2 = (real, imag)

        if self.normalize:
            means = torch.mean(complex_specs, dim=[1, 2, 3], keepdim=True)
            std = torch.std(complex_specs, dim=[1, 2, 3], keepdim=True)
            complex_specs = (complex_specs - means) / (std + 1e-8)

        ### processing
        x = complex_specs
        identity = complex_specs

        x = self.first_conv(x)
        x, encoder_outputs = self.encoder(x)
        x = self.decoder(x, encoder_outputs)
        mask = self._mask_processing(x)

        if identity.shape[2] != mask.shape[2]:
            h_diff = abs(identity.shape[2] - mask.shape[2])
            identity = identity[:, :, :-h_diff, :]
        if identity.shape[3] != mask.shape[3]:
            w_diff = abs(identity.shape[3] - mask.shape[3])
            identity = identity[:, :, :, :-w_diff]

        clean_estimate_spec = identity * mask

        ### istft
        batch, complex_channel, freq, time = clean_estimate_spec.shape
        clean_estimate_spec = clean_estimate_spec.view(batch, complex_channel*freq, time)

        clean_estimate_wav = self.istft(clean_estimate_spec)
        clean_estimate_wav = torch.clamp_(clean_estimate_wav, -1, 1)
        return clean_estimate_spec, clean_estimate_wav

    def _mask_processing(self, x, eps=1e-8, method='bounded_tanh'):
        if method == 'unbounded':
            mask = x
        elif method == 'bounded_sigmoid':
            mask = torch.sigmoid(x)
        elif method == 'bounded_tanh':
            mask_mag = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)
            mask_phase = x / mask_mag
            mask = mask_mag * mask_phase * torch.tanh(mask_mag)

        return mask

if __name__ == '__main__':
    window_size = 1024  # 16000Hz * 0.064s (=64ms) in paper
    hop_size = 256  # 16000Hz * 0.016s (=16ms) in paper
    fft_size = 1024  # same as window_size

    model = DCUNet('dcunet20-large', window_size, hop_size, fft_size, is_complex=True)
    
    x = torch.rand(1, 32000)
    clean_estimate_spec, clean_estimate_wav = model(x)
    print(clean_estimate_wav.shape)