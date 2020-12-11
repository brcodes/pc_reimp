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
    # sizes of representations and weights
    input_size = attr.ib(default=256,validator=attr.validators.instance_of(int))
    output_size = attr.ib(default=10,validator=attr.validators.instance_of(int))
    hidden_sizes = attr.ib(default=[32],validator=attr.validators.instance_of(list))
    # hidden layer variance
    sigma = attr.ib(default=[1.0],validator=attr.validators.instance_of(list))
    # priors on parameters
    alpha = attr.ib(default=1.0,validator=attr.validators.instance_of(float)) #" related to variance of Gaussian priors "
    lam = attr.ib(default=0.02,validator=attr.validators.instance_of(float)) #" related to variance of Gaussian priors "
    r_prior = attr.ib(default="gaussian",validator=attr.validators.in_(['gaussian','kurtotic']))
    U_prior = attr.ib(default="gaussian",validator=attr.validators.in_(['gaussian','kurtotic']))
    # unit activation function (linear, tanh, relu, etc.)
    unit_act = attr.ib(default="linear",validator=attr.validators.in_(['linear','tanh']))
    # learning schedules - these should become strings for a learning schedule dispatcher
    k_r = attr.ib(default=0.5,validator=attr.validators.instance_of(float))
    k_U = attr.ib(default=0.5,validator=attr.validators.instance_of(float))
    # training time
    batch_size = attr.ib(default=10,validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(default=100,validator=attr.validators.instance_of(int))
