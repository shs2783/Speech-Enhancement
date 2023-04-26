'https://arxiv.org/pdf/2104.05267.pdf'

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_stft import ConvSTFT, ConviSTFT

class ConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))

class DeConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
        super().__init__()

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return self.conv_transpose1(x) * torch.sigmoid(self.conv_transpose2(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, gate=False, **kwargs) -> None:
        super().__init__()

        if gate:
            self.conv = ConvGLU(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.PReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, gate=False, **kwargs) -> None:
        super().__init__()

        if gate:
            self.conv_transposed = DeConvGLU(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        else:
            self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.PReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv_transposed(x)))


class Attention(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_u, x_c):
        identity = x_c

        x_u = self.conv1(x_u)
        x_c = self.conv2(x_c)
        x = torch.sigmoid(x_u + x_c)

        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x * identity

class Encoder(nn.Module):
    def __init__(self, in_channels=2, gate=False) -> None:
        super().__init__()

        encoder_channels = [16, 32, 64, 96, 128, 128]
        self.layers = nn.ModuleList()
        for out_channels in encoder_channels:
            self.layers += [ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), gate=gate)]
            in_channels = out_channels

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs

class Decoder(nn.Module):
    def __init__(self, in_channels=128, gate=False) -> None:
        super().__init__()

        decoder_channels = [128, 96, 64, 32, 16, 2]
        self.conv_transpose_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        for i, out_channels in enumerate(decoder_channels):
            self.attention_layers += [Attention(in_channels)]
            self.conv_transpose_layers += [ConvTransposeBlock(in_channels*2, out_channels, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), output_padding=(1, 0), gate=gate)]
            in_channels = out_channels

    def forward(self, x, encoder_outputs):
        for i in range(len(self.conv_transpose_layers)):
            encoder_output = encoder_outputs.pop()
            if x.shape[2] < encoder_output.shape[2]:
                x = F.pad(x, (0, 0, 0, 1))

            x = self.attention_layers[i](x, encoder_output)
            x = torch.cat([x, encoder_output], dim=1)
            x = self.conv_transpose_layers[i](x)

        return x


class CARN(nn.Module):
    def __init__(self, window_size=320, hop_size=160, fft_size=512, lstm_channels=512, gate=False) -> None:
        super().__init__()

        self.fft_size = fft_size
        self.stft = ConvSTFT(window_size, hop_size, fft_size)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        self.encoder = Encoder(in_channels=2, gate=gate)
        self.decoder = Decoder(in_channels=128, gate=gate)

        self.lstm = nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=fft_size, out_features=fft_size + 2)

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

        x, (h, c) = self.lstm(x)
        x = x.permute(0, 2, 1).view(batch, channels, freq, time)

        x = self.decoder(x, encoder_outputs)

        x = x.view(batch, channels*freq, time).permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1).view(batch, 2, self.fft_size//2 + 1, time)

        ### mask processing
        mask_real = x[:, 0, :, :]
        mask_imag = x[:, 1, :, :]

        real = mask_real * noisy_real - mask_imag * noisy_imag
        imag = mask_real * noisy_imag - mask_imag * noisy_real

        ### istft
        clean_estimate_spec = torch.cat([real, imag], dim=1)
        clean_estimate_wav = self.istft(clean_estimate_spec)
        clean_estimate_wav = torch.clamp_(clean_estimate_wav, -1, 1)
        return clean_estimate_spec, clean_estimate_wav

class GCARN(CARN):
    def __init__(self, window_size=320, hop_size=160, fft_size=512, lstm_channels=512) -> None:
        super(GCARN, self).__init__(window_size, hop_size, fft_size, lstm_channels, gate=True)

if __name__ == '__main__':
    signal = torch.randn(1, 16000)
    # model = CARN(window_size=320, hop_size=160, fft_size=512, lstm_channels=512)
    model = GCARN(window_size=320, hop_size=160, fft_size=512, lstm_channels=512)

    clean_estimate_spec, clean_estimate_wav = model(signal)
    print(clean_estimate_wav.shape)