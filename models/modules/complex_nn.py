import torch
import torch.nn as nn

def complex_concat(inputs, dim=1):
    real_list = []
    imag_list = []

    for x in inputs:
        real, imag = torch.chunk(x, 2, dim=dim)
        real_list.append(real)
        imag_list.append(imag)

    real = torch.cat(real_list, dim=dim)
    imag = torch.cat(imag_list, dim=dim)
    output = torch.cat([real, imag], dim=dim)
    return output

def split_complex(x, dim=1):
    '''return real and imag part of input'''

    if torch.is_complex(x):             # (batch, complex channels, height, width)
        real, imag = x.real, x.imag
    elif isinstance(x, (tuple, list)):  # [real tensor, imag tensor]  ## tensor shape = (batch, channels, height, width)
        real, imag = x
    elif isinstance(x, torch.Tensor):   # (batch, real + imag channels, height, width)
        real, imag = torch.chunk(x, 2, dim=dim)
    else:
        raise ValueError("Input must be a complex tensor or a tuple of real and imaginary tensors")
    
    return real, imag

def merge_real_imag(x, real, imag, dim=1):
    '''return output in the same format as input'''

    if isinstance(x, (tuple, list)):
        output = [real, imag]
    elif isinstance(x, torch.Tensor):
        output = torch.cat([real, imag], dim=dim)
    elif torch.is_complex(x):
        output = torch.complex(real, imag)
    
    return output
    

class ComplexConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.real_conv = None
        self.imag_conv = None

    def forward(self, x):
        real, imag = split_complex(x)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        output = merge_real_imag(x, real, imag)
        return output

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

class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()

        assert in_channels % 2 == 0, f'in_channels must be a factor of 2, current channels: {in_channels}'
        assert out_channels % 2 == 0, f'out_channels must be a factor of 2, current channels: {out_channels}'

        in_channels = in_channels//2
        out_channels = out_channels//2
        
        self.real_linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.imag_linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        real, imag = split_complex(x, dim=-1)

        real = self.real_linear(real)
        imag = self.imag_linear(imag)

        output = merge_real_imag(x, real, imag, dim=-1)
        return output

class ComplexLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, **kwargs) -> None:
        super().__init__()

        assert in_channels % 2 == 0, f'in_channels must be a factor of 2, current channels: {in_channels}'
        assert hidden_channels % 2 == 0, f'hidden_channels must be a factor of 2, current channels: {hidden_channels}'

        in_channels = in_channels//2
        hidden_channels = hidden_channels//2
        
        self.real_lstm = nn.LSTM(in_channels, hidden_channels, **kwargs)
        self.imag_lstm = nn.LSTM(in_channels, hidden_channels, **kwargs)

    def forward(self, x):
        real, imag = split_complex(x, dim=-1)

        real2real, (real2real_h, real2real_c) = self.real_lstm(real)
        imag2imag, (imag2imag_h, imag2imag_c) = self.imag_lstm(imag)

        real2imag, (real2imag_h, real2imag_c) = self.imag_lstm(real)
        imag2real, (imag2real_h, imag2real_c) = self.real_lstm(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        output = merge_real_imag(x, real, imag, dim=-1)
        return output
        
    def flatten_parameters(self):
        self.real_lstm.flatten_parameters()
        self.imag_lstm.flatten_parameters()


class ComplexBatchNorm2d:  # TODO: implement
    ''' https://arxiv.org/pdf/1705.09792.pdf '''

class ComplexReLU:  # TODO: implement
    ''' https://arxiv.org/pdf/1705.09792.pdf '''

class ComplexLeakyReLU:  # TODO: implement
    ...

ComplexBatchNorm2d = nn.BatchNorm2d
ComplexReLU = nn.ReLU
ComplexLeakyReLU = nn.LeakyReLU