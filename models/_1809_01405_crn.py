'https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf'
'https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement'

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        alpha = kwargs.get('alpha', 1)
        self.padding = kwargs.get('padding', (0, 0))
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ELU(alpha) if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding[0], :]  # causal
        x = self.act(self.norm(x))
        return x
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        alpha = kwargs.get('alpha', 1)
        self.padding = kwargs.get('padding', (0, 0))
        
        self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ELU(alpha) if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv_transposed(x)
        x = x[:, :, :-1, :]  # causal
        x = self.act(self.norm(x))
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1) -> None:
        super().__init__()

        encoder_channels = [16, 32, 64, 128, 256]
        self.layers = nn.ModuleList()
        for out_channels in encoder_channels:
            self.layers += [ConvBlock(in_channels, out_channels, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))]
            in_channels = out_channels
    
    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs

class Decoder(nn.Module):
    def __init__(self, in_channels=512) -> None:
        super().__init__()
        
        decoder_channels = [128, 64, 32, 16, 1]
        self.layers = nn.ModuleList()
        for i, out_channels in enumerate(decoder_channels):
            if i == 3:
                self.layers += [ConvTransposeBlock(in_channels, out_channels, kernel_size=(2, 3), stride=(1, 2), output_padding=(0, 1))]
            elif i == 4:
                self.layers += [ConvTransposeBlock(in_channels, out_channels, kernel_size=(2, 3), stride=(1, 2), norm=False, act=False)]
            else:
                self.layers += [ConvTransposeBlock(in_channels, out_channels, kernel_size=(2, 3), stride=(1, 2))]
            in_channels = out_channels * 2
    
    def forward(self, x, encoder_outputs):
        for layer in self.layers:
            encoder_output = encoder_outputs.pop()
            x = torch.cat([x, encoder_output], dim=1)
            x = layer(x)
        return x
    
class CRN(nn.Module):
    def __init__(self, window_size=320, hop_size=160, fft_size=320) -> None:
        super().__init__()

        self.stft = ConvSTFT(window_size, hop_size, fft_size, return_mag_phase=True)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        self.encoder = Encoder()
        self.lstm_layers = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)
        self.decoder = Decoder()
        
    def forward(self, x):
        mag, phase = self.stft(x)  # (batch, freq, time)
        mag = mag.transpose(1, 2).unsqueeze(1)  # (batch, channel, time, freq)

        x, encoder_outputs = self.encoder(mag)
        batch, channel, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, channel*freq)

        x, _ = self.lstm_layers(x)
        x = x.reshape(batch, time, channel, freq).permute(0, 2, 1, 3)

        x = self.decoder(x, encoder_outputs)
        x = F.softplus(x)

        clean_estimate_spec = x.squeeze(1).transpose(1, 2)  # (batch, freq, time)
        clean_estimate_wav = self.istft(clean_estimate_spec, phase)
        return clean_estimate_spec, clean_estimate_wav
    
if __name__ == '__main__':
    window_size = 320
    hop_size = 160
    fft_size = 320  # inference from input size frequency 161

    model = CRN(window_size, hop_size, fft_size)

    signal = torch.randn(1, 32000)
    clean_estimate_spec, clean_estimate_wav = model(signal)  # (batch, channel, time, freq)
    print(clean_estimate_spec.shape, clean_estimate_wav.shape)