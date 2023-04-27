'https://arxiv.org/pdf/1807.06514.pdf'

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

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels, bias=not norm, **kwargs)
        self.norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.linear(x)))
    
class ChannelAttention(nn.Module):
    def __init__(self, feature_map_channels, r=16) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = LinearBlock(feature_map_channels, feature_map_channels//r)
        self.linear2 = LinearBlock(feature_map_channels//r, feature_map_channels)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(batch_size, channels, 1, 1)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, feature_map_channels, r=16, d=4) -> None:
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2 * d
        
        self.layers = nn.Sequential(
            ConvBlock(feature_map_channels, feature_map_channels//r, kernel_size=1),
            ConvBlock(feature_map_channels//r, feature_map_channels//r, kernel_size, padding=padding, dilation=d),
            ConvBlock(feature_map_channels//r, feature_map_channels//r, kernel_size, padding=padding, dilation=d),
            ConvBlock(feature_map_channels//r, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class BAM(nn.Module):
    def __init__(self, feature_map_channels, reduction=16, dilation=4) -> None:
        super().__init__()
        
        self.channel_attention_branch = ChannelAttention(feature_map_channels, reduction)
        self.spatial_attention_branch = SpatialAttention(feature_map_channels, reduction, dilation)
    
    def forward(self, x):
        channel_attention = self.channel_attention_branch(x)  # (batch_size, channels, 1, 1)
        spatial_attention = self.spatial_attention_branch(x)  # (batch_size, 1, height, width)
        attention_map = torch.sigmoid(channel_attention + spatial_attention)  # broadcast operation
        return x + (x * attention_map)
    
if __name__ == "__main__":
    feature_map_channels = 256
    reduction = 16
    dilation = 4
    model = BAM(feature_map_channels, reduction, dilation)
    
    x = torch.randn(10, feature_map_channels, 224, 224)
    y = model(x)
    print(y.shape)