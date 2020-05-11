import torch.optim.lr_scheduler as lr_scheduler


class scheduler(object):
    @staticmethod
    def StepLR(step_size, gamma=0.1, last_epoch=-1):
        return lambda x: lr_scheduler.StepLR(x, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def MultiStepLR(milestones, gamma=0.1, last_epoch=-1):
        return lambda x: lr_scheduler.MultiStepLR(x, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    @staticmethod
    def ReduceLROnPlateau(mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                          threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        return lambda x: lr_scheduler.ReduceLROnPlateau(x, mode=mode, factor=factor, patience=patience, verbose=verbose,
                                                        threshold=threshold, threshold_mode=threshold_mode,
                                                        cooldown=cooldown, min_lr=min_lr, eps=eps)

    @staticmethod
    def step(scheduler, epoch_loss):
        if isinstance(scheduler, lr_scheduler.StepLR):
            scheduler.step()
        elif isinstance(scheduler, lr_scheduler.MultiStepLR):
            scheduler.step()
        elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
