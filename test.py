import sys
sys.path.append('models')

import torch
from utils import show_params
from models.dcunet import DCUNet

if __name__ == '__main__':
    is_complex = True
    model = DCUNet('dcunet20', is_complex)
    show_params(model)
    
    channels = 2 if is_complex else 1
    x = torch.randn(1, channels, 256, 800)
    y = model(x)
    print(y.shape)