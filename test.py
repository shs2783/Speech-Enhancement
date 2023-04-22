import sys
sys.path.append('models')

import torch
from utils import show_params
from models import CRN

if __name__ == '__main__':
    model = CRN()
    show_params(model)
    
    x = torch.randn(1, 1, 100, 161)
    y = model(x)
    print(y.shape)