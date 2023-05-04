'https://arxiv.org/pdf/2102.01993.pdf'

import torch
import torch.nn as nn
from .complex_nn import ComplexConv2d, ComplexLinear, ComplexBatchNorm2d, split_complex, merge_real_imag, complex_concat

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, act=True, **kwargs) -> None:
        super().__init__()
        
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, bias=not norm, **kwargs)
        self.norm = ComplexBatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=True, **kwargs) -> None:
        super().__init__()
        
        self.linear = ComplexLinear(in_channels, out_channels, **kwargs)
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.linear(x))
    
class ChannelAttention(nn.Module):
    def __init__(self, feature_map_channels, r=16) -> None:
        super().__init__()

        if feature_map_channels // r == 0:
            reduction_channels = 2
        else:
            reduction_channels = feature_map_channels // r
            
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.shared_fc_layer = nn.Sequential(
            LinearBlock(feature_map_channels, reduction_channels, act=True, bias=False),
            LinearBlock(reduction_channels, feature_map_channels, act=False, bias=False)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape  # channels = feature map channels * 2
        real, imag = split_complex(x)
        
        avg_real = self.avg_pool(real).flatten(start_dim=1)
        avg_imag = self.avg_pool(imag).flatten(start_dim=1)
        
        max_real = self.max_pool(real).flatten(start_dim=1)
        max_imag = self.max_pool(imag).flatten(start_dim=1)
        
        avg_x = torch.cat([avg_real, avg_imag], dim=1)
        max_x = torch.cat([max_real, max_imag], dim=1)
        
        avg_x = self.shared_fc_layer(avg_x)
        max_x = self.shared_fc_layer(max_x)
        
        channel_attention_map = torch.sigmoid(avg_x + max_x)
        channel_attention_map = channel_attention_map.view(batch_size, channels, 1, 1)
        return channel_attention_map
    
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv = ConvBlock(in_channels=4, out_channels=2, kernel_size=7, padding=3)
        
    def forward(self, x):
        real, imag = split_complex(x)
        
        avg_real = torch.mean(real, dim=1, keepdim=True)
        avg_imag = torch.mean(imag, dim=1, keepdim=True)
        
        max_real = torch.max(real, dim=1, keepdim=True)[0]
        max_imag = torch.max(imag, dim=1, keepdim=True)[0]
        
        avg_x = torch.cat([avg_real, avg_imag], dim=1)
        max_x = torch.cat([max_real, max_imag], dim=1)
        
        pooled_feature_map = complex_concat([avg_x, max_x], dim=1)
        spatial_attention_map = self.conv(pooled_feature_map)
        spatial_attention_map = torch.sigmoid(spatial_attention_map)
        return spatial_attention_map

class CCBAM(nn.Module):
    def __init__(self, feature_map_channels, reduction=16) -> None:
        super().__init__()
        
        self.channel_attention_branch = ChannelAttention(feature_map_channels, reduction)
        self.spatial_attention_branch = SpatialAttention()
    
    def forward(self, x):
        channel_attention = self.channel_attention_branch(x)  # (batch_size, channels, 1, 1)
        x *= channel_attention  # broadcast operation
        
        spatial_attention = self.spatial_attention_branch(x)  # (batch_size, 2, height, width)  ## 2 = (real, imag)
        
        real, imag = split_complex(x)
        real = real + spatial_attention[:, 0, :, :].unsqueeze(1)  # broadcast operation
        imag = imag + spatial_attention[:, 1, :, :].unsqueeze(1)  # broadcast operation
        x = merge_real_imag(x, real, imag, dim=1)

        return x
    
if __name__ == "__main__":
    feature_map_channels = 256 * 2  # feature map channels * 2 (complex)
    reduction = 16
    model = CCBAM(feature_map_channels, reduction)
    
    x = torch.randn(10, feature_map_channels, 224, 224)
    y = model(x)
    print(y.shape)