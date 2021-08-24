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
    input_size = attr.ib(default=576,validator=attr.validators.instance_of(int))
    output_size = attr.ib(default=10,validator=attr.validators.instance_of(int))
    hidden_sizes = attr.ib(default=[96,32],validator=attr.validators.instance_of(list))
    # tiling
    tile_offset = attr.ib(default=6,validator=attr.validators.instance_of(int))
    # hidden layer variance - NOTE: must have >= number of elements as hidden_sizes
    # r&b have layer 1 and 2 values as 1.0, 10.0 respectively; the final value is from mli pc model "r3"
    sigma_sq = attr.ib(default={1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0, 9: 2.0},validator=attr.validators.instance_of(dict))
    # priors on parameters (layer 1 and 2 have r&b values, the rest are from mli pc model)
    alpha = attr.ib(default={1: 1.0, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05},validator=attr.validators.instance_of(dict)) #" related to variance of Gaussian priors "
    # layer 1 has r&b value, the rest are from mli pc model
    lam = attr.ib(default={1: 0.02, 2: 0.00001, 3: 0.00001, 4: 0.00001, 5: 0.00001, 6: 0.00001, 7: 0.00001},validator=attr.validators.instance_of(dict)) #" related to variance of Gaussian priors "
    r_prior = attr.ib(default="gaussian",validator=attr.validators.in_(['gaussian','kurtotic']))
    U_prior = attr.ib(default="gaussian",validator=attr.validators.in_(['gaussian','kurtotic']))
    # unit activation function (linear, tanh, relu, etc.)
    unit_act = attr.ib(default="linear",validator=attr.validators.in_(['linear','tanh']))
    # classification cost type
    classification = attr.ib(default="NC",validator=attr.validators.in_(['NC','C1','C2']))
    # classifcation cost boosting parameter
    c_cost_param = attr.ib(default=1,validator=attr.validators.instance_of(int))
    # learning rate schedulers
    k_r_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.0005}},validator=attr.validators.instance_of(dict))
    k_U_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.005}},validator=attr.validators.instance_of(dict))
    k_o_sched = attr.ib(kw_only=True,default={'constant':{'initial':0.0005}},validator=attr.validators.instance_of(dict))
    # KEEPING THESE FOR NOW, BUT EVENTUALLY THEY WILL BE GONE
    # learning schedules - these should become strings for a learning schedule dispatcher
    #k_r = attr.ib(default=0.0005,validator=attr.validators.instance_of(float))
    #k_U = attr.ib(default=0.005,validator=attr.validators.instance_of(float))
    #k_o = attr.ib(default=0.5,validator=attr.validators.instance_of(float))
    # training time
    batch_size = attr.ib(default=1,validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(default=10,validator=attr.validators.instance_of(int))
