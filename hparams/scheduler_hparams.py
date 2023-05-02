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