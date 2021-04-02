import numpy as np

def constant_lr(epoch, initial):
    '''
    Simply clamps the learning rate at the initial values forever.
    '''
    return initial


def step_decay_lr(epoch, initial, drop_factor, drop_every):
    '''
    Drops the learning rate by a fixed factor every drop_factor epochs.
    '''
    exp_fac = np.floor((1+epoch)/drop_every)
    return initial*np.power(drop_factor,exp_fac)


def polynomial_decay_lr(epoch, initial, max_epochs, poly_power):
    '''
    Drops the learning rate to zero over max_epochs epochs, with
    shape given by poly_pow (set poly_pow = 1 to get linear decay).
    '''
    decay = np.power((1 - (epoch/max_epochs)),poly_power)
    return initial*decay
