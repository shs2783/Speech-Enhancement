'https://arxiv.org/pdf/1505.04597.pdf'

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        self.conv_transposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv_transposed(x)))

class Encoder(nn.Module):
    def __init__(self, channels=64, repeat_num=4) -> None:
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(repeat_num):
            layers += [nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                ConvBlock(channels*(2**i), channels*(2**(i+1)), kernel_size=3),
                ConvBlock(channels*(2**(i+1)), channels*(2**(i+1)), kernel_size=3),
            )]
        
        self.layers = layers

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            outputs.append(x)
            x = layer(x)

        return x, outputs

class Decoder(nn.Module):
    def __init__(self, channels=1024, repeat_num=4) -> None:
        super().__init__()
        
        layers = nn.ModuleList()
        for i in range(repeat_num):
            layers += [
                ConvTransposeBlock(channels//(2**i), channels//(2**(i+1)), kernel_size=2, stride=2),
                ConvBlock(channels//(2**i), channels//(2**(i+1)), kernel_size=3),
                ConvBlock(channels//(2**(i+1)), channels//(2**(i+1)), kernel_size=3),
            ]
        
        self.layers = layers

    def forward(self, x, encoder_outputs):
        for layer in self.layers:
            x = layer(x)
            
            if isinstance(layer, ConvTransposeBlock):
                output = encoder_outputs.pop()
                left = (output.shape[-1] - x.shape[-1]) // 2
                right = left + x.shape[-1]
                cropped = output[:, :, left:right, left:right]
                x = torch.cat([x, cropped], dim=1)
            
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_repeat=4, encoder_channels=64, decoder_channels=1024) -> None:
        super().__init__()
        
        self.first_conv1 = ConvBlock(in_channels, encoder_channels, kernel_size=3)
        self.first_conv2 = ConvBlock(encoder_channels, encoder_channels, kernel_size=3)
        self.encoder = Encoder(encoder_channels, num_repeat)
        self.decoder = Decoder(decoder_channels, num_repeat)
        self.last_conv = nn.Conv2d(encoder_channels, out_channels=2, kernel_size=1)
        
    def forward(self, x):
        x = self.first_conv1(x)
        x = self.first_conv2(x)
        
        x, encoder_outputs = self.encoder(x)
        x = self.decoder(x, encoder_outputs)
        
        x = self.last_conv(x)
        return x

if __name__ == '__main__':
    model = UNet()
    
    x = torch.randn(1, 1, 572, 572)
    y = model(x)
    print(y.shape)