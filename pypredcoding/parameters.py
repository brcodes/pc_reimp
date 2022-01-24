import attr,os

def file_not_found(instance,attribute,value):
    '''
    Checks if a file exists; used as a custom validator for parameters that
    should be files or paths, checking to be sure you set them to things that
    actually exist.
    '''
    if not os.path.exists(value):
        raise ValueError('{} does not exist!'.format(attribute.name))


@attr.s(kw_only=True)
class ModelParameters(object):
    '''
    Class for holding and checking parameters relevant to model creation.
    '''
    # Sizes of representations and weights
    input_size = attr.ib(default=16384,validator=attr.validators.instance_of(int))
    hidden_sizes = attr.ib(default=[96,128],validator=attr.validators.instance_of(list))
    output_size = attr.ib(default=5,validator=attr.validators.instance_of(int))
    # Tiling (if num_r1_mods > 1: model will respond as if it receiving tiled input)
    num_r1_mods = attr.ib(default=225,validator=attr.validators.instance_of(int))
    # Unit activation function (linear, tanh, relu, etc.)
    act_fxn = attr.ib(default="lin",validator=attr.validators.in_(['lin','tan']))
    # Priors
    r_prior = attr.ib(default="gaus",validator=attr.validators.in_(['gaus','kurt']))
    U_prior = attr.ib(default="gaus",validator=attr.validators.in_(['gaus','kurt']))
    # Classification cost scheme
    class_scheme = attr.ib(default="c1",validator=attr.validators.in_(['nc','c1','c2']))
    # Training time
    batch_size = attr.ib(default=1,validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(default=10,validator=attr.validators.instance_of(int))
    # Learning rate schedulers
    k_r_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.0005}},validator=attr.validators.instance_of(dict))
    k_U_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.005}},validator=attr.validators.instance_of(dict))
    k_o_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.0005}},validator=attr.validators.instance_of(dict))
    # Hidden layer variance (each attr dict must have >= number of keys:values as hidden_sizes)
    # RB99 have layer 1 and 2 values as 1.0, 10.0 respectively; the final value of 2.0 is from Li PC model "r3"
    sigma_sq = attr.ib(default={1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0, 9: 2.0},validator=attr.validators.instance_of(dict))
    # Parameters on priors (layer 1 and 2 have RB99 values, the rest are from Li PC model)
    alpha = attr.ib(default={1: 1.0, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05},validator=attr.validators.instance_of(dict)) #" related to variance of Gaussian priors "
    # Layer 1 has RB99 value, the rest are from Li PC model
    lam = attr.ib(default={1: 0.02, 2: 0.00001, 3: 0.00001, 4: 0.00001, 5: 0.00001, 6: 0.00001, 7: 0.00001},validator=attr.validators.instance_of(dict)) #" related to variance of Gaussian priors "
