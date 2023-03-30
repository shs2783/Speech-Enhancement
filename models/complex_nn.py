import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.real_conv = None
        self.imag_conv = None
        
    def forward(self, x):
        if isinstance(x, (tuple, list)):   # [real tensor, imag tensor]  ## tensor shape = (batch, channels, height, width)
            real, imag = x
        elif isinstance(x, torch.Tensor):  # (batch, real + imag channels, height, width)
            real, imag = torch.chunk(x, 2, dim=1)
        elif torch.is_complex(x):          # (batch, complex channels, height, width)
            real, imag = x.real, x.imag
        else:
            raise ValueError("Input must be a complex tensor or a tuple of real and imaginary tensors")
        
        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)
        
        real = real2real - imag2imag
        imag = real2imag + imag2real

        if isinstance(x, (tuple, list)):
            return [real, imag]
        elif isinstance(x, torch.Tensor):
            return torch.cat([real, imag], dim=1)
        elif torch.is_complex(x):
            return torch.complex(real, imag)

class ComplexConv2d(ComplexConv):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        assert in_channels % 2 == 0, f'in_channels must be a factor of 2, current channels: {in_channels}'
        assert out_channels % 2 == 0, f'out_channels must be a factor of 2, current channels: {out_channels}'
        
        in_channels = in_channels//2  
        out_channels = out_channels//2
        
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    
class ComplexConvTranspose2d(ComplexConv):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        assert in_channels % 2 == 0, f'in_channels must be a factor of 2, current channels: {in_channels}'
        assert out_channels % 2 == 0, f'out_channels must be a factor of 2, current channels: {out_channels}'
        
        in_channels = in_channels//2
        out_channels = out_channels//2
        
        self.real_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
        
        
class ComplexBatchNorm2d:  # TODO: implement
    ''' https://arxiv.org/pdf/1705.09792.pdf '''

class ComplexReLU:  # TODO: implement
    ''' https://arxiv.org/pdf/1705.09792.pdf '''

class ComplexLeakyReLU:  # TODO: implement
    ...

ComplexBatchNorm2d = nn.BatchNorm2d
ComplexReLU = nn.ReLU
ComplexLeakyReLU = nn.LeakyReLU