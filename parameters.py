import attr,os

def file_not_found(instance,attribute,value):
    '''
    Checks is a file exists; used as a custom validator for parameters that
    should be files or paths, checking to be sure you set them to things that
    actually exist.
    '''
    if not os.path.exists(value):
        raise ValueError('{} does not exist!'.format(attribute.name))


@attr.s
class ModelParameters(object):
    '''
    Class for holding and checking parameters relevant to model creation.
    '''
    input_size = attr.ib(kw_only=True,validator=attr.validators.instance_of(int))
    num_layers = attri.ib(kw_only=True,default=2,validator=attr.validators.instance_of(int))
