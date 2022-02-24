import attr,os

def file_not_found(instance,attribute,value):
    '''
    Checks if a file exists; used as a custom validator for parameters that
    should be files or paths, checking to be sure you set them to things that
    actually exist.
    '''
    if not os.path.exists(value):
        raise ValueError('{} does not exist!'.format(attribute.name))

def size_params_to_p_format(num_nonin_lyrs=3, lyr_sizes=(96,128,5), num_imgs=5, numxpxls=128, numypxls=128):

        ### Automatically set the remainder of model parameters for parameters (p) object creation, loading
        # Defaults are Li classification model params

        ## Image input size
        input_size = numxpxls * numypxls

        ## Hidden layer sizes
        hidden_sizes = []
        for hl in range(0,num_nonin_lyrs-1):
            hidden_sizes.append(lyr_sizes[hl])

        ## Model output size (last layer)
        output_size = lyr_sizes[-1]
        if output_size != num_imgs:
            print("Output size (output_size) must == number of input images (num_imgs)")
            exit()

        return input_size, hidden_sizes, output_size

def LR_params_to_dict(lr_scheme="constant", r_init=0.005, U_init=0.01, o_init=0.0005,
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

    elif lr_scheme == "step":
        k_r_sched[lr_scheme]["drop_factor"] = r_drop_factor
        k_U_sched[lr_scheme]["drop_factor"] = U_drop_factor
        k_o_sched[lr_scheme]["drop_factor"] = o_drop_factor

        k_r_sched[lr_scheme]["drop_every"] = r_drop_every
        k_U_sched[lr_scheme]["drop_every"] = U_drop_every
        k_o_sched[lr_scheme]["drop_every"] = o_drop_every

    return k_r_sched, k_U_sched, k_o_sched


@attr.s(kw_only=True)
class ModelParameters(object):
    '''
    Class for holding and checking parameters relevant to model creation.
    '''
    # Sizes of representations and weights
    input_size = attr.ib(default=16384,validator=attr.validators.instance_of(int))
    # First element in hidden_sizes is the size of only one single layer 1 module
    # If tiled data input into model, sum of all layer 1 module params becomes hidden_sizes[0] * numtiles (Li case: 32*225: 7200)
    hidden_sizes = attr.ib(default=[32,128],validator=attr.validators.instance_of(list))
    output_size = attr.ib(default=5,validator=attr.validators.instance_of(int))
    # Tiling (if num_r1_mods > 1: model will respond as if it receiving tiled input)
    num_r1_mods = attr.ib(default=225,validator=attr.validators.instance_of(int))
    # Unit activation function (linear, tanh, relu, etc.)
    act_fxn = attr.ib(default="lin",validator=attr.validators.in_(['lin','tan']))
    # Priors
    r_prior = attr.ib(default="kurt",validator=attr.validators.in_(['gaus','kurt']))
    U_prior = attr.ib(default="kurt",validator=attr.validators.in_(['gaus','kurt']))
    # Gradient update scheme (r update equilibration before U update [RB99] or 30 simultaneous r,U updates [Li])
    update_scheme = attr.ib(default="rU_simultaneous",validator=attr.validators.in_(["rU_simultaneous", "r_eq_then_U"]))
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
    # Checkpointing during training; def is checkpoint every 10 epochs
    checkpointing = attr.ib(default=["every_n_ep",10],validator=attr.validators.instance_of(list))
