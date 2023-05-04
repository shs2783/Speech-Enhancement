'https://arxiv.org/pdf/1807.06521v2.pdf'

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
    def __init__(self, in_channels, out_channels, act=True, **kwargs) -> None:
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
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
        batch_size, channels, height, width = x.shape
        
        avg_x = self.avg_pool(x).flatten(start_dim=1)
        max_x = self.max_pool(x).flatten(start_dim=1)
        
        avg_x = self.shared_fc_layer(avg_x)
        max_x = self.shared_fc_layer(max_x)
        
        channel_attention_map = torch.sigmoid(avg_x + max_x)
        channel_attention_map = channel_attention_map.view(batch_size, channels, 1, 1)
        return channel_attention_map
    
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv = ConvBlock(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        
        pooled_feature_map = torch.cat([avg_x, max_x], dim=1)
        spatial_attention_map = self.conv(pooled_feature_map)
        spatial_attention_map = torch.sigmoid(spatial_attention_map)
        return spatial_attention_map

class CBAM(nn.Module):
    def __init__(self, feature_map_channels, reduction=16) -> None:
        super().__init__()
        
        self.channel_attention_branch = ChannelAttention(feature_map_channels, reduction)
        self.spatial_attention_branch = SpatialAttention()
    
    def forward(self, x):
        channel_attention = self.channel_attention_branch(x)  # (batch_size, channels, 1, 1)
        x *= channel_attention  # broadcast operation
        
        spatial_attention = self.spatial_attention_branch(x)  # (batch_size, 1, height, width)
        x *= spatial_attention  # broadcast operation
        return x
    
if __name__ == "__main__":
    feature_map_channels = 256
    reduction = 16
    model = CBAM(feature_map_channels, reduction)
    
    x = torch.randn(10, feature_map_channels, 224, 224)
    y = model(x)
    print(y.shape)