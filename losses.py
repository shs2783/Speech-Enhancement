import torch
import torch.nn as nn

def SDR_loss(estimate, target):
    '''
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    output_norm = torch.norm(target - estimate, dim=1) ** 2
    target_norm = torch.norm(target, dim=1) ** 2
    sdr = 10 * torch.log10(target_norm / output_norm)
    return -torch.mean(sdr)

def SDR_loss2(estimate, target):
    '''
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    output_norm = torch.norm(estimate, dim=1) ** 2
    target_norm = torch.sum(target*estimate, dim=1) ** 2
    sdr = 10 * torch.log10(target_norm / output_norm)
    return -torch.mean(sdr)

def modified_SDR_loss(estimate, target):
    output_norm = torch.norm(estimate, dim=1)
    target_norm = torch.norm(target, dim=1)
    dot_product = torch.sum(target*estimate, dim=1)
    return -dot_product / output_norm * target_norm

def Weighted_SDR_loss(noisy_signal, estimate, target):
    '''
    noisy_signal: (batch_size, signal_length)
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    clean_estimate = estimate
    clean_target = target
    noise_estimate = noisy_signal - estimate
    noise_target = noisy_signal - target

    sdr_clean = modified_SDR_loss(clean_estimate, clean_target)
    sdr_noise = modified_SDR_loss(noise_estimate, noise_target)

    clean_target_ns = torch.square(torch.norm(clean_target, dim=1))  # norm squared
    noise_target_ns = torch.square(torch.norm(noise_target, dim=1))  # norm squared
    alpha = clean_target_ns / (clean_target_ns + noise_target_ns + 1e-8)

    weighted_sdr = alpha * sdr_clean + (1 - alpha) * sdr_noise
    return -torch.mean(weighted_sdr)

def SI_SNR_loss(estimate, target, zero_mean=False):
    '''
    output: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    if zero_mean:
        estimate -= torch.mean(estimate)
        target -= torch.mean(target)

    target_norm = torch.norm(target, dim=1, keepdim=True)
    output_target_dot_product = torch.sum(estimate * target, dim=1, keepdim=True)

    s_target = (output_target_dot_product * target) / target_norm ** 2
    s_target_norm = torch.norm(s_target, dim=1)
    s_target_norm_squared = s_target_norm**2 + 1e-8

    e_noise = estimate - s_target
    e_noise_norm = torch.norm(e_noise, dim=1)
    e_noise_norm_squared = e_noise_norm**2 + 1e-8

    snr = 10 * torch.log10(s_target_norm_squared / e_noise_norm_squared)
    return -torch.mean(snr)

if __name__ == '__main__':
    x = torch.randn(2, 10)
    x1 = torch.randn(2, 10)
    x2 = torch.randn(2, 10)

    si_snr = SI_SNR_loss(x1, x2)
    sdr1 = SDR_loss(x1, x2)
    weighted_sdr = Weighted_SDR_loss(x, x1, x2)
    print(si_snr, sdr1, weighted_sdr)