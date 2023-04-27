'https://arxiv.org/pdf/1609.07132.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_stft import ConvSTFT, ConviSTFT
from architectures import ced_architecture

class LazyConvBlock(nn.Module):
    '''I am too lazy to calculate decoder input size'''

    def __init__(self, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        self.conv = nn.LazyConv1d(out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, encoder_architecture, use_max_pool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for x in encoder_architecture:
            out_channels, kernel_size = x
            padding = (kernel_size - 1) // 2

            self.layers += [nn.Sequential(
                LazyConvBlock(out_channels, kernel_size=kernel_size, padding=padding),
                nn.MaxPool1d(kernel_size=2, stride=2) if use_max_pool else nn.Identity()
            )]

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return x, outputs

class Decoder(nn.Module):
    def __init__(self, decoder_architecture, use_upsample=False) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.use_upsample = use_upsample

        for x in decoder_architecture:
            out_channels, kernel_size = x
            padding = (kernel_size - 1) // 2

            self.layers += [nn.Sequential(
                LazyConvBlock(out_channels, kernel_size=kernel_size, padding=padding),
                nn.Upsample(scale_factor=2, mode='nearest') if use_upsample else nn.Identity()
            )]
    
    def forward(self, x, encoder_outputs):
        for i, layer in enumerate(self.layers):
            if self.use_upsample:
                encoder_output = encoder_outputs.pop()
                x = torch.cat([x, encoder_output], dim=1)
                x = layer(x)

            else:
                x = layer(x)
                if i > 0:
                    encoder_output = encoder_outputs.pop()
                    x = torch.cat([x, encoder_output], dim=1)
                
        if self.use_upsample:
            x = F.pad(x, (2, 1))  # to match the input size
        return x

class CED(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        architecture = ced_architecture['ced']
        enc_length = len(architecture) // 2

        encoder_architecture = architecture[:enc_length]
        decoder_architecture = architecture[enc_length:]

        self.encoder = Encoder(encoder_architecture, use_max_pool=True)
        self.decoder = Decoder(decoder_architecture, use_upsample=True)
        self.final_conv = nn.LazyConv1d(out_channels=1, kernel_size=129, padding=64)
    
    def forward(self, x):
        x, encoder_outputs = self.encoder(x)
        x = self.decoder(x, encoder_outputs)
        x = self.final_conv(x)
        return x

class RCED(nn.Module):
    def __init__(self, config='r-ced10', cascaded=False) -> None:
        super().__init__()
        architecture = ced_architecture[config]
        enc_length = len(architecture) // 2

        encoder_architecture = architecture[:enc_length]
        decoder_architecture = architecture[enc_length:]

        self.encoder = Encoder(encoder_architecture)
        self.decoder = Decoder(decoder_architecture)

        self.cascaded = cascaded
        if not cascaded:
            self.final_conv = nn.LazyConv1d(out_channels=1, kernel_size=129, padding=64)

    def forward(self, x):
        x, encoder_outputs = self.encoder(x)
        x = self.decoder(x, encoder_outputs)

        if not self.cascaded:
            x = self.final_conv(x)
        return x

class CRCED(nn.Module):
    def __init__(self, repeat_num=5) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            *[RCED('cr-ced', cascaded=True) for _ in range(repeat_num)]
        )
        self.final_conv = nn.LazyConv1d(out_channels=1, kernel_size=129, padding=64)

    def forward(self, x):
        x = self.layers(x)
        x = self.final_conv(x)
        return x
    
if __name__ == '__main__':
    window_size = 256  # 8000Hz * 0.032s (=32ms) in paper
    hop_size = 64  # 8000Hz * 0.008s (=8ms) in paper
    fft_size = 256

    stft = ConvSTFT(window_size, hop_size, fft_size, return_mag_phase=True)
    istft = ConviSTFT(window_size, hop_size, fft_size)

    # model = CED()
    # model = RCED('r-ced10')
    model = CRCED()

    signal = torch.randn(1, 448)
    mag, phase = stft(signal)  # (batch, freq, time)
    mag = mag.transpose(1, 2)  # (batch, time, freq)
    print(mag.shape)

    output = model(mag)  # (batch, channel, time, freq)
    print(output.shape)