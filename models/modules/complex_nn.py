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

    if isinstance(x, (tuple, list)):   # [real tensor, imag tensor]  ## tensor shape = (batch, channels, height, width)
        real, imag = x
    elif torch.is_complex(x):          # (batch, complex channels, height, width)
        real, imag = x.real, x.imag
    elif isinstance(x, torch.Tensor):  # (batch, real + imag channels, height, width)
        real, imag = torch.chunk(x, 2, dim=dim)
    else:
        raise ValueError("Input must be a complex tensor or a tuple of real and imaginary tensors")
    
    return real, imag

def merge_real_imag(x, real, imag, dim=1):
    '''return output in the same format as input'''

    if isinstance(x, (tuple, list)):
        output = [real, imag]
    elif torch.is_complex(x):
        output = torch.complex(real, imag)
    elif isinstance(x, torch.Tensor):
        output = torch.cat([real, imag], dim=dim)
    
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


class ComplexBatchNorm2d(nn.Module):
    '''
    https://arxiv.org/pdf/1705.09792.pdf
    https://github.com/huyanxin/DeepComplexCRN/blob/master/complexnn.py
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, complex_axis=1):
        super().__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):
        # self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class ComplexReLU:  # TODO: implement
    ''' https://arxiv.org/pdf/1705.09792.pdf '''

class ComplexLeakyReLU:  # TODO: implement
    ...

class ComplexPReLU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.real_prelu = nn.PReLU(**kwargs)
        self.imag_prelu = nn.PReLU(**kwargs)

    def forward(self, x):
        real, imag = split_complex(x, dim=1)

        real2real = self.real_prelu(real)
        imag2imag = self.imag_prelu(imag)

        real2imag = self.imag_prelu(real)
        imag2real = self.real_prelu(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        output = merge_real_imag(x, real, imag, dim=1)
        return output

ComplexReLU = nn.ReLU
ComplexLeakyReLU = nn.LeakyReLU