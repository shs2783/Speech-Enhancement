import torch
import torch.nn.functional as F

def SDR_loss(estimate, target, eps=1e-8):
    '''
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    target_norm = torch.norm(target, dim=1)
    output_norm = torch.norm(target - estimate, dim=1)

    target_norm = target_norm ** 2 + eps
    output_norm = output_norm ** 2 + eps

    sdr = 10 * torch.log10(target_norm / output_norm)
    return torch.mean(sdr)

def SDR_loss2(estimate, target, eps=1e-8):
    '''
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    target_norm = torch.sum(target*estimate, dim=1)
    output_norm = torch.norm(estimate, dim=1)

    target_norm = target_norm ** 2 + eps
    output_norm = output_norm ** 2 + eps

    sdr = 10 * torch.log10(target_norm / output_norm)
    return torch.mean(sdr)

def modified_SDR_loss(estimate, target):
    output_norm = torch.norm(estimate, dim=1)
    target_norm = torch.norm(target, dim=1)
    dot_product = torch.sum(target*estimate, dim=1)
    return -dot_product / output_norm * target_norm

def Weighted_SDR_loss(noisy_signal, estimate, target, eps=1e-8):
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
    alpha = clean_target_ns / (clean_target_ns + noise_target_ns + eps)

    weighted_sdr = alpha * sdr_clean + (1 - alpha) * sdr_noise
    return torch.mean(weighted_sdr)

def SI_SNR_loss(estimate, target, zero_mean=False, eps=1e-8):
    '''
    estimate: (batch_size, signal_length)
    target: (batch_size, signal_length)
    '''

    if zero_mean:
        estimate -= torch.mean(estimate)
        target -= torch.mean(target)

    target_norm = torch.norm(target, dim=1, keepdim=True)
    output_target_dot_product = torch.sum(estimate * target, dim=1, keepdim=True)

    s_target = (output_target_dot_product * target) / target_norm ** 2
    s_target_norm = torch.norm(s_target, dim=1)
    s_target_norm = s_target_norm ** 2 + eps

    e_noise = estimate - s_target
    e_noise_norm = torch.norm(e_noise, dim=1)
    e_noise_norm = e_noise_norm ** 2 + eps

    snr = 10 * torch.log10(s_target_norm / e_noise_norm)
    return torch.mean(snr)

def mask_loss(mask, x, y):
    '''
    mask : estimate mask (batch_size, freq, time)
    x : noisy spectrogram (batch_size, freq, time)
    y : clean spectrogram (batch_size, freq, time)
    '''
    
    x_real, x_imag = torch.chunk(x, 2, dim=1)
    y_real, y_imag = torch.chunk(y, 2, dim=1)
    
    ground_mask_real = (x_real * y_real + x_imag * y_imag) / (x_real**2 + x_imag**2)
    ground_mask_imag = (x_real * y_imag - x_imag * y_real) / (x_real**2 + x_imag**2)
    ground_mask = torch.cat([ground_mask_real, ground_mask_imag], dim=1)
    
    loss = F.mse_loss(mask, ground_mask)
    return loss

if __name__ == '__main__':
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    estimate = torch.randn(2, 16000)

    si_snr = SI_SNR_loss(estimate, clean)
    sdr1 = SDR_loss(estimate, clean)
    sdr2 = SDR_loss2(estimate, clean)
    weighted_sdr = Weighted_SDR_loss(noisy, estimate, clean)
    print(si_snr, sdr1, sdr2, weighted_sdr)
    
    mask = torch.randn(2, 64, 10)
    noisy_spec = torch.randn(2, 64, 10)
    clean_spec = torch.randn(2, 64, 10)
    loss = mask_loss(mask, noisy_spec, clean_spec)
    print(loss)