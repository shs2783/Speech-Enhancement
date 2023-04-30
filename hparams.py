from multiprocessing import cpu_count

class HyperParams:
    # training
    batch_size = 16
    num_epochs = 100
    lr = 1e-3
    patience = 5
    current_epoch = 0
    no_improvement = 0
    best_loss = 1e10
    clip_norm = 0.5
    seed = 2023

    # dataset
    mix_audio_path = 'synth/mix'
    split_audio_path = ['synth/clean', 'synth/noise']
    k = 5  # only use when mix_audio_path is None
    sr = 16000

    # dataloader
    chunk_size = 32000
    pin_memory = True,
    num_workers = cpu_count() // 2

    # model
    window_size = 512
    hop_size = 256
    fft_size = 512
    lstm_channels = 512

    # checkpoint dir
    checkpoint = 'checkpoint'
    load_model_path = 'checkpoint/best_model.pt'


class OptimizerHparams:
    @staticmethod
    def sgd(lr=0.1, momentum=0.9, weight_decay=0, nesterov=False):
        return dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    @staticmethod
    def rmsprop(lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        return dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

    @staticmethod
    def adadelta(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        return dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

    @staticmethod
    def adagrad(lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        return dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)

    @staticmethod
    def adamax(lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        return dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @staticmethod
    def adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        return dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    @staticmethod
    def adamw(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        return dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    @staticmethod
    def sparse_adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        return dict(lr=lr, betas=betas, eps=eps)

    @staticmethod
    def nadam(lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004):
        return dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)


class SchedulerHparams:
    @staticmethod
    def lambda_lr(lr_lambda, last_epoch=-1):
        return dict(lr_lambda=lr_lambda, last_epoch=last_epoch)

    @staticmethod
    def multiplicative_lr(lr_lambda, last_epoch=-1):
        return dict(lr_lambda=lr_lambda, last_epoch=last_epoch)

    @staticmethod
    def step_lr(step_size=30, gamma=0.1, last_epoch=-1):
        return dict(step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def multi_step_lr(milestones=[30, 80], gamma=0.1, last_epoch=-1):
        return dict(milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    @ staticmethod
    def constant_lr(last_epoch=-1):
        return dict(last_epoch=last_epoch)

    @staticmethod
    def linear_lr(lr_lambda, last_epoch=-1):
        return dict(lr_lambda=lr_lambda, last_epoch=last_epoch)

    @staticmethod
    def exponential_lr(gamma=0.1, last_epoch=-1):
        return dict(gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def polynomial_lr(max_epoch, power=1.0, last_epoch=-1):
        return dict(max_epoch=max_epoch, power=power, last_epoch=last_epoch)

    @staticmethod
    def cosine_annealing_lr(T_max=50, eta_min=0, last_epoch=-1):
        return dict(T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)

    @staticmethod
    def reduce_lr_on_plateau(mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        return dict(mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)

    @staticmethod
    def cyclic_lr(base_lr=0.001, max_lr=0.01, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        return dict(base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode=mode, gamma=gamma, scale_fn=scale_fn, scale_mode=scale_mode, cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum, last_epoch=last_epoch)

    @staticmethod
    def one_cycle_lr(max_lr=0.01, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=-1):
        return dict(max_lr=max_lr, total_steps=total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start, anneal_strategy=anneal_strategy, cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum, div_factor=div_factor, final_div_factor=final_div_factor, three_phase=three_phase, last_epoch=last_epoch)

    @staticmethod
    def cosine_annealing_warm_restarts(T_0=50, T_mult=1, eta_min=0, last_epoch=-1):
        return dict(T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)