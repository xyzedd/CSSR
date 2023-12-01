

import os
import numpy as np
import torch

img_size = 32


def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_lr(opts, lr):
    for op in opts:
        for param_group in op.param_groups:
            param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmUpLrSchedule:

    def __init__(self, warm_epoch, epoch_tot_steps, init_lr):
        self.ep_steps = epoch_tot_steps
        self.tgtstep = warm_epoch * epoch_tot_steps
        self.init_lr = init_lr
        self.warm_epoch = warm_epoch

    def get_lr(self, epoch, step, lr):
        tstep = epoch * self.ep_steps + step
        if self.tgtstep > 0 and tstep <= self.tgtstep:
            lr = self.init_lr * tstep / self.tgtstep
        return lr


class MultiStepLrSchedule:

    def __init__(self, milestones, lrdecays, start_lr, warmup_schedule=None):
        super().__init__()
        self.milestones = milestones
        self.warmup = warmup_schedule
        self.lrdecays = lrdecays
        self.start_lr = start_lr

    # step 表示epoch中已经输入过的样本数
    def get_lr(self, epoch, step, lr):
        lr = self.start_lr
        # if step == 0 : # update learning rate
        for m in self.milestones:
            if epoch >= m:
                lr *= self.lrdecays
        # print("LEARNRATE",lr)
        if self.warmup is not None:
            lr = self.warmup.get_lr(epoch, step, lr)
        # print("LEARNRATE",lr)
        return lr

# cosine_s,cosine_e = 0,0


# epoch wise
class EpochwiseCosineAnnealingLrSchedule:

    def __init__(self, startlr, milestones, lrdecay, epoch_num, warmup=None):
        super().__init__()
        self.cosine_s, self.cosine_e = 0, 0
        self.milestones = milestones
        self.lrdecay = lrdecay
        self.warmup = warmup
        self.warmup_epoch = 0 if warmup is None else warmup.warm_epoch
        self.epoch_num = epoch_num
        self.startlr = startlr
        self.ms = [self.warmup_epoch] + self.milestones + [self.epoch_num]
        self.ref = {self.ms[i]: self.ms[i+1] for i in range(len(self.ms)-1)}

    def get_lr(self, epoch, step, lr):
        #global cosine_s,cosine_e
        if self.warmup is not None:
            lr = self.warmup.get_lr(epoch, step, lr)
        if step != 0:
            return lr
        if epoch in self.ms:
            if epoch != self.warmup_epoch:
                self.startlr *= self.lrdecay
            self.cosine_s = epoch
            self.cosine_e = self.ref[epoch]
        #print("calc lr",epoch,self.ms,self.cosine_s,self.cosine_e)
        if self.cosine_e > 0:
            lr = self.startlr * \
                (np.cos((epoch - self.cosine_s) /
                 (self.cosine_e - self.cosine_s) * 3.14159)+1) * 0.5

        return lr


# Step wise
class StepwiseCosineAnnealingLrSchedule:

    def __init__(self, startlr, epoch_tot_steps, milestones, lrdecay, epoch_num, warmup=None):
        super().__init__()
        self.cosine_s, self.cosine_e = 0, 0
        self.milestones = milestones
        self.lrdecay = lrdecay
        self.warmup = warmup
        self.warmup_epoch = 0 if warmup is None else warmup.warm_epoch
        self.epoch_num = epoch_num
        self.startlr = startlr
        self.ms = [self.warmup_epoch] + self.milestones + [self.epoch_num]
        self.ref = {self.ms[i]: self.ms[i+1] for i in range(len(self.ms)-1)}
        self.ep_steps = epoch_tot_steps

    # step wise
    def get_lr(self, epoch, step, lr):
        if self.warmup is not None:
            lr = self.warmup(epoch, step, lr)
        if step == 0 and epoch in self.ms:
            if epoch != self.warmup_epoch:
                self.startlr *= self.lrdecay
            self.cosine_s = epoch
            self.cosine_e = self.ref[epoch]
        if self.cosine_e > 0:
            steps = step + (epoch - self.cosine_s) * self.epoch_tot_steps
            lr = self.startlr * \
                (np.cos(steps / (self.cosine_e - self.cosine_s) /
                 self.epoch_tot_steps * 3.14159)+1) * 0.5
        return lr


def get_scheduler(config, train_loader):
    if config['lr_schedule'] == 'multi_step':
        warmup = WarmUpLrSchedule(config['warmup_epoch'], len(
            train_loader), config['learn_rate'])
        return MultiStepLrSchedule(config["milestones"], config['lr_decay'], config['learn_rate'], warmup)
    elif config['lr_schedule'] == 'cosine':
        warmup = WarmUpLrSchedule(config['warmup_epoch'], len(
            train_loader), config['learn_rate'])
        return EpochwiseCosineAnnealingLrSchedule(config['learn_rate'], config["milestones"], config['lr_decay'], config['epoch_num'], warmup)


method_list = {}


class regmethod:

    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, func, *args, **kwds):
        global method_list
        method_list[self.name] = func
        print("Registering", self.name)
        return func
