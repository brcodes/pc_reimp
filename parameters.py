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
    input_size = attr.ib(validator=attr.validators.instance_of(int))
    num_layers = attr.ib(default=2,validator=attr.validators.instance_of(int))
    layer_size =
    batch_size = attr.ib(default=10,validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(default=100,validator=attr.validators.instance_of(int))
    #" this is the learning rate.  It could be a constant, or a function that depends on epoch. (We can talk about this.)"
    k = attr.ib(default=0.5,validator=attr.validators.instance_of(float))
    # this is a number (set of numbers), one per layer, assumed and not computed
    sigma = attr.ib(default=1.0,validator=attr.validators.instance_of(float))
    alpha = attr.ib(default=1.0,validator=attr.validators.instance_of(float)) #" related to variance of Gaussian priors "
    lam = attr.ib(default=0.02,validator=attr.validators.instance_of(float)) #" related to variance of Gaussian priors "
