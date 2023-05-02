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

