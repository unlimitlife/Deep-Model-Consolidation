import torch.optim as optim


class optimizer(object):
    @staticmethod
    def SGD(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        return lambda x: optim.SGD(x, lr=lr, momentum=momentum, dampening=dampening,
                                   weight_decay=weight_decay, nesterov=nesterov)
    def Adam(lr, betas=(0.9,0.999)):
        return lambda x: optim.Adam(x, lr=lr, betas=betas)
