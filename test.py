import sys
sys.path.append('models')

import torch
from utils import show_params
from models import CED, RCED, CRCED

if __name__ == '__main__':
    # model = CED()
    # model = RCED('r-ced16')
    model = CRCED()
    
    x = torch.randn(1, 8, 129)
    y = model(x)
    print(y.shape)
    
    show_params(model)