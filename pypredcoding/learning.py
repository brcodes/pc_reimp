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

def LR_params_to_dict(r_init=0.005, U_init=0.01, o_init=0.0005,
    r_max_eps=500, U_max_eps=500, o_max_eps=500,
    r_poly_power=1, U_poly_power=1, o_poly_power=1,
    r_drop_factor=1, U_drop_factor=0.98522, o_drop_factor=1,
    r_drop_every=40, U_drop_every=40, o_drop_every=40):

        ### Logic to automatically arrange LR parameters dict for p object

        k_r_sched = {lr_scheme:{"initial":r_init}}
        k_U_sched = {lr_scheme:{"initial":U_init}}
        k_o_sched = {lr_scheme:{"initial":o_init}}

        if lr_scheme == "constant":
            pass

        elif lr_scheme == "poly":
            k_r_sched[lr_scheme]["max_epochs"] = r_max_eps
            k_U_sched[lr_scheme]["max_epochs"] = U_max_eps
            k_o_sched[lr_scheme]["max_epochs"] = o_max_eps

            k_r_sched[lr_scheme]["poly_power"] = r_poly_power
            k_U_sched[lr_scheme]["poly_power"] = U_poly_power
            k_o_sched[lr_scheme]["poly_power"] = o_poly_power

        # If lr_scheme == "step"
        else:
            k_r_sched[lr_scheme]["drop_factor"] = r_drop_factor
            k_U_sched[lr_scheme]["drop_factor"] = U_drop_factor
            k_o_sched[lr_scheme]["drop_factor"] = o_drop_factor

            k_r_sched[lr_scheme]["drop_every"] = r_drop_every
            k_U_sched[lr_scheme]["drop_every"] = U_drop_every
            k_o_sched[lr_scheme]["drop_every"] = o_drop_every

        return k_r_sched, k_U_sched, k_o_sched
