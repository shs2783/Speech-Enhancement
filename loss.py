import torch
import torch.nn as nn

def SDR_loss(output, target):
    '''
    output: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''
    
    output_norm = output / torch.norm(output, dim=1, keepdim=True)
    target_norm = target / torch.norm(target, dim=1, keepdim=True)
    sdr = torch.sum(output_norm * target_norm, dim=1)
    return torch.mean(sdr)

def SI_SNR_loss(output, target):
    '''
    output: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''
    
    target_norm = torch.norm(target, dim=1, keepdim=True)
    output_target_dot_product = torch.sum(output * target, dim=1)
    
    s_target = (output_target_dot_product * target) / torch.sum(target_norm**2, dim=1)
    s_target_norm = torch.norm(s_target, dim=1, keepdim=True)
    sum_squared_s_target = torch.sum(s_target_norm**2, dim=1) + 1e-8
    
    e_noise = output - target
    e_noise_norm = torch.norm(e_noise, dim=1, keepdim=True)
    sum_squared_e_noise_norm = torch.sum(e_noise_norm**2, dim=1) + 1e-8
    snr = 10 * torch.log10(sum_squared_s_target / sum_squared_e_noise_norm)
    return torch.mean(snr)