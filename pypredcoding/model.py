from cost_functions import StaticCostFunction, RecurrentCostFunction
import numpy as np
from functools import partial

from datetime import datetime
import pickle
from os import makedirs
from os.path import join, exists, dirname, isfile
import csv

class PredictiveCodingClassifier:

    def __init__(self):
        
        # Choices for transformation functions, priors
        self.act_fxn_dict = {'linear': self.linear_transform,
                                'tanh': self.tanh_transform}
        self.prior_cost_dict = {'gaussian': self.gaussian_prior_costs, 
                                'kurtotic': self.kurtotic_prior_costs,
                                'Li_priors': self.Li_prior_costs}
        
        self.r_prior_dist_dict = {'gaussian': partial(np.random.normal, loc=0, scale=1),
                                'kurtotic': partial(np.random.laplace, loc=0.0, scale=0.5),
                                'Li_priors': self.create_zeros}
        
        self.U_prior_dist_dict = {'gaussian': partial(np.random.normal, loc=0, scale=1),
                                'kurtotic': partial(np.random.laplace, loc=0.0, scale=0.5),
                                'Li_priors': self.create_Li_rand}
        
        '''
        shell class for sPCC and rPCC subclasses
        inheritance: some general methods, and all attributes
        
        "self."
        
        # metadata and experiment (this is grabbed by run_experiment, but saved here as a flag)
        mod_name: str: name of chosen model if eval, pred or end-state model if train
        exp_name: str: name of experiment
        train_with: bool: whether to train the model (in here, has or has not init a training)
        evaluate_with: bool: whether to evaluate the model (in here, has or has not init an evaluation)
        predict_with: bool: whether to predict with the model (in here, has or has not init a prediction)
        notes: str: any notes about the experiment
        
        # model
        model_type: str: 'static' or 'recurrent'
        tiled: bool: whether to tile the input data
        flat_input: bool: whether to flatten the input data
        num_layers: int: number of layers in the model (discluding input layer '0')
        input_size: tuple: size of input data, one sample
        hidden_lyr_sizes: list: size of hidden layers
        output_lyr_size: int: size of output layer (if c1, must == num_classes)
        classif_method: str or None: 'c1' or 'c2' or None
        activ_func: str: 'linear' or 'tanh'
        priors: str: 'gaussian' or 'kurtotic'
        update_method: dict: {'rW_niters':#} or {'r_niters_W':#} or {'r_eq_W':#} # is int for iters, float for eq
        
        # dataset train
        num_imgs: int: number of images in dataset
        num_classes: int: number of classes in dataset
        dataset_train: str: name of dataset to grab from data/
        
        # training
        batch_size: int: number of samples in each batch
        epoch_n: int: number of epochs to train
        kr: dict: learning rates for each layer, r component
        kU: dict: learning rates for each layer, U component
        kV: dict: learning rates for each layer, V component
        alph: dict: prior parameter for each layer, r component
        lam: dict: prior parameter for each layer, U component
        ssq: dict: layer var/covar parameter for each layer
        
        # training data
        save_checkpoint: dict or None: {'save_every':#} or {'fraction':#} # is int, num or denom
        load_checkpoint: int or None: number of checkpoint to load. -1 for most recent. None for no load
        online_diagnostics: bool: whether to save and print diagnostics during training (loss, accuracy)
        plot_train: bool: whether to plot training diagnostics. online_diagnostics must be True
        
        # dataset evaluation
        dataset_eval: str: name of dataset to grab from data/
        
        # evaluation data
        plot_eval: str or None: 'first' for first image, etc. (see model.py)
        
        # dataset prediction
        dataset_pred: str: name of dataset to grab from data/
        
        # prediction data
        plot_pred: str or None: 'first' for first image, etc. (see model.py)
        '''
        
    def set_model_attributes(self, params):
        '''
        Set model attributes from a dictionary.
        This will set a bunch of external attributes too,
        which will serve no other purpose than to recount the last experiment run on the model. e.g. name, train, notes.
        '''
        for key, value in params.items():
            setattr(self, key, value)
        
    def config_from_attributes(self):
        '''
        Set up the model from the attributes.
        '''
        
        self.validate_attributes()
        
        # Transforms and priors
        self.f = self.act_fxn_dict[self.activ_func]
        self.g = self.prior_cost_dict[self.priors]
        self.h = self.prior_cost_dict[self.priors]
        
        self.r = {}
        self.U = {}
        
        
        '''
        Li style: test
        '''
        
        # Initiate rs, Us (Vs in recurrent subclass)
        # r0 is going to be different
        tiled_input = self.tiled_input
        flat_input = self.flat_input
        architecture = self.architecture
        input_shape = self.input_shape
        num_layers = self.num_layers
        n = num_layers
        
        '''
        scenario
        
        tiling IS receptive fields.
        No tiling removes that.
        
        tiled: True, flat: True    Input (e.g.) 16,864    U1 16,864,32    32 r1  (FHL)          U2 32,128
                                                16,864    U1 16,864,32    16,32 r1 (E1L_Li)     U2 512,128
                                                16,864    U1 16,864,32    16,32 r1 (E1L)        U2 16,32,128
        
        tiled: True, flat: False   I (e.g.) 4,4,24,36     U1 4,4,24,36,32    32 r1  (FHL)       U2 32,128
                                            4,4,24,36     U1 4,4,24,36,32    4,4,32 r1 (E1L)    U2 4,4,32,128
                                            
        tiled: False, flat: True   I (e.g.) 1,11088       U1 1,11088,32     32 r1  (all)        U2 32,128
                                            
        tiled: False, flat: False  I (e.g.) 84,132     U1 84,132,32    32 r1  (all)             U2 32,128
        
        '''
        # In all cases, U1 is input dims plus r1 (hidden layer sizes [0])
        # U2 is r1dims plus r2 dims 
        #                                       (except EIL_li which is r1dims.flatten() plus r2 dims)
        # U3 - n is r1dims plus r2 dims
        
        # If flat hidden layers, r1 is hidden layer sizes [0]
        
        # Initiate rs and Us
        # r0
        self.r[0] = np.zeros(input_shape)
        for i in range(1, n + 1):
            # Layer 1
            if i == 1:
                # r1
                if architecture == 'flat_hidden_layers':
                    self.r[1] = self.r_prior_dist(size=(self.hidden_lyr_sizes[0]))
                # Li architecture is only meant for one type of input
                elif architecture == 'expand_first_lyr_Li':
                    if tiled_input and flat_input:
                        self.r[1] = self.r_prior_dist(size=(self.num_tiles, self.hidden_lyr_sizes[0]))
                        
                elif architecture == 'expand_first_lyr':
                    if tiled_input and flat_input:
                        self.r[1] = self.r_prior_dist(size=(self.num_tiles, self.hidden_lyr_sizes[0]))
                    elif tiled_input and not flat_input:
                        self.r[1] = self.r_prior_dist(size=(self.input_shape[0], self.input_shape[1], self.hidden_lyr_sizes[0]))
                    elif not tiled_input:
                        self.r[1] = self.r_prior_dist(size=(self.hidden_lyr_sizes[0]))
                # U1
                input_shape_list = list(input_shape)
                input_shape_list.append(self.hidden_lyr_sizes[0])
                U1_size = tuple(input_shape_list)
                print(f'U1_size: {U1_size}')
                self.U[1] = self.U_prior_dist(size=U1_size)
                
            # Layer i > 1 and i < n
            elif i > 1 and i < n:
                # ri
                self.r[i] = self.r_prior_dist(size=(self.hidden_lyr_sizes[i - 1]))
                # U2
                if i == 2:
                    if architecture == 'expand_first_lyr_Li':
                        if tiled_input and flat_input:
                            U2_size = (self.r[1].shape[0] * self.r[1].shape[1], self.hidden_lyr_sizes[i - 1])
                            self.U[2] = self.U_prior_dist(size=U2_size)
                    else:
                        # Unpack all the dimension sizes in r1
                        U2_size = (*self.r[1].shape, self.r[2].shape[0])
                        self.U[2] = self.U_prior_dist(size=U2_size)
                # U3+
                else:
                    Ui_size = (self.r[i-1].shape[0], self.r[i].shape[0])
                    self.U[i] = self.U_prior_dist(size=Ui_size)
            # Layer n
            elif i == n:
                # rn
                self.r[n] = self.r_prior_dist(size=(self.output_lyr_size))
                # Un
                Un_size = (self.r[n-1].shape[0], self.r[n].shape[0])
                self.U[n] = self.U_prior_dist(size=Un_size)
                
        # Classification now
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.output_lyr_size)
            self.U['o'] = self.U_prior_dist(size=Uo_size)
        
            
        # Initiate U1-based operations dims (dimentions flex based on input)
        # Transpose dims
        ndims_U1 = len(U1_size)
        range_ndims_U1 = range(ndims_U1)
        last_dim_id_U1 = range_ndims_U1[-1]
        nonlast_dim_ids_U1 = range_ndims_U1[:-1]
        transpose_dims_U1 = tuple([last_dim_id_U1] + list(nonlast_dim_ids_U1))
        self.U1T_dims = transpose_dims_U1
        
        # Tensordot dims
        # U1T, last n-1
        nonfirst_dim_ids_U1 = range_ndims_U1[1:]
        self.U1T_tdot_dims = list(nonfirst_dim_ids_U1)
        
        # Input dims, all
        ndims_input = len(input_size)
        range_ndims_input = range(ndims_input)
        # Bottom-up error and cost dims (either used in update, or cost)
        self.bu_error_tdot_dims = list(range_ndims_input)
        
        # Einsum dims
        self.einsum_arg_U1 = ''
        dim_str = 'ijklmnopqrstuvwxyz'
        for dim in range(ndims_input):
            self.einsum_arg_U1 += dim_str[dim]
        self.einsum_arg_U1 += ',' + dim_str[ndims_input] + '->' + dim_str[:ndims_input + 1]
        if ndims_input > len(dim_str):
            raise ValueError('Too many dimensions.')
        # Will always be 'i,j->ij' for U2 through Un
        self.einsum_arg_Ui = 'i,j->ij'
        
        # Hidden layer sizes for priors
        self.all_hlyr_sizes = self.hidden_lyr_sizes.copy()
        self.all_hlyr_sizes.append(self.output_lyr_size)
        if tiled:
            self.all_hlyr_sizes[0] = (self.num_tiles, self.all_hlyr_sizes[0])
            
        
        '''
        Li style: test
        '''
        
        # Initiate Jr, Jc, and accuracy (diagnostics) for storage, print, plot
        epoch_n = self.epoch_n
        self.Jr = [0] * (epoch_n + 1)
        self.Jc = [0] * (epoch_n + 1)
        self.accuracy = [0] * (epoch_n + 1)
        
        # Later: by layer
        # self.Jr = {i: [0] * (epoch_n + 1) for i in range(n)}
        # self.Jc = {i: [0] * (epoch_n + 1) for i in range(n)}
        
    def set_component_update_funcs(self, num_layers, model_type, classif_method):
        n = num_layers
            
        if n == 1:
            rkey = Ukey = 1
        elif n == 2:
            rkey = Ukey = 2
        elif n >= 3:
            rkey = 3
            Ukey = 2
            
        keys = [rkey, Ukey]
            
        self.r_updates = self.r_upd_func_dict[keys[0]]
        self.U_updates = self.U_upd_func_dict[keys[1]]
        
        self.component_updates = [self.r_updates, self.U_updates]
        
        if classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
            
        if model_type == 'recurrent':
            Vkey = 1
            keys.append(Vkey)
            self.V_updates = self.V_upd_func_dict[keys[2]]
            self.component_updates.append(self.V_updates)
            
        
        
        
        
        
        
        classif_cost = self.classif_cost_dict[self.classif_method]
        # Component update: r, U, Uo, and cost calculato
        n = num_layers
        if n == 1:
            self.r_updates = self.r_updates_n_1
            self.U_updates = self.U_updates_n_1
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_1
        elif n == 2:
            self.r_updates = self.r_updates_n_2
            self.U_updates = self.U_updates_n_gt_eq_2
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_2
        elif n >= 3:
            self.r_updates = self.r_updates_n_gt_eq_3
            self.U_updates = self.U_updates_n_gt_eq_2
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
        self.component_updates = [self.r_updates, self.U_updates]
        classif_method = self.classif_method
        if classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
        # Add Vs
        self.component_updates.append(self.V_updates)
        
        pass
        
    # Activation functions
    def linear_transform(self, U_dot_r):
        """
        Though intended to operate on some U.dot(r), will take any numerical
        argument x and return the tuple (f(x), F(x)). Linear transformation.
        """
        f = U_dot_r
        if f.ndim == 1:
            F = np.eye(len(f))
        elif f.ndim == 2:
            F = np.eye(f.shape[0], f.shape[1])
        else:
            shape = f.shape
            F = np.eye(shape[0], shape[1])
            for dim in range(2, len(shape)):
                F = np.expand_dims(F, axis=dim)
                F = np.repeat(F, shape[dim], axis=dim)
        return (f, F)
    
    def tanh_transform(self, U_dot_r):
        """
        Though intended to operate on some U.dot(r), will take any numerical
        argument x and return the tuple (f(x), F(x)). Tanh transformation.
        """
        f = np.tanh(U_dot_r)
        F = np.diag(1 - f.flatten()**2)
        return (f, F)
    
    # r, U or V prior functions
    def gaussian_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian random prior.
        
        Catch overflow here.
        """
        printlog = self.print_and_log
        
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            func_eval = alph_or_lam * np.square(r_or_U).sum()
            func_deriv_eval = 2 * alph_or_lam * r_or_U
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (func_eval, func_deriv_eval)
        
        except FloatingPointError as e:
            printlog(f"FloatingPointError: {e}")
            printlog(f'Overflow is checked in prior cost evaluation, and has been encountered.')
            
            if r_or_U is not None:
                printlog("r or U shape:", r_or_U.shape)
                # Create a slice object that slices the first 5 elements in each dimension
                slice_obj = tuple(slice(0, 3) for _ in range(r_or_U.ndim))
                printlog("A few elements of r_or_U:\n", r_or_U[slice_obj])
            
            return None
    
    def kurtotic_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
        
        Catch overflow here.
        """
        printlog = self.print_and_log
        
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            func_eval = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
            func_deriv_eval = (2 * alph_or_lam * r_or_U) / (1 + np.square(r_or_U))
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (func_eval, func_deriv_eval)
        
        except FloatingPointError as e:
            printlog(f"FloatingPointError: {e}")
            printlog(f'Overflow is checked in prior cost evaluation, and has been encountered.')
            
            if r_or_U is not None:
                printlog("r or U shape:", r_or_U.shape)
                # Create a slice object that slices the first 5 elements in each dimension
                slice_obj = tuple(slice(0, 3) for _ in range(r_or_U.ndim))
                printlog("A few elements of r_or_U:\n", r_or_U[slice_obj])
            
            return None
    

    def Li_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Li 'kurtotic' prior.
        
        Catch overflow here.
        """
        printlog = self.print_and_log
        
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            func_eval = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
            func_deriv_eval = (alph_or_lam * r_or_U) / (1 + np.square(r_or_U))
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (func_eval, func_deriv_eval)
        
        except FloatingPointError as e:
            printlog(f"FloatingPointError: {e}")
            printlog(f'Overflow is checked in prior cost evaluation, and has been encountered.')
            
            if r_or_U is not None:
                printlog("r or U shape:", r_or_U.shape)
                # Create a slice object that slices the first 5 elements in each dimension
                slice_obj = tuple(slice(0, 3) for _ in range(r_or_U.ndim))
                printlog("A few elements of r_or_U:\n", r_or_U[slice_obj])
            
            return None
        
    def create_zeros(self, size):
        return np.zeros(size)

    def create_Li_rand(self, size):
        # Shifted random uniform 
        return np.random.rand(*size) - 0.5

    def r_prior_dist(self, size):
        return self.r_prior_dist_dict[self.priors](size=size)
    
    def U_prior_dist(self, size):
        return self.U_prior_dist_dict[self.priors](size=size)
    
    def hard_set_r_prior_dist(self):
        '''
        so a new one isn't made every image
        timesaver, but not a truly random draw
        '''
        
        self.r_dists_hard = {}
        n = self.num_layers
        for i in range (1, n + 1):
            lyr_size = self.all_hlyr_sizes[i - 1]
            self.r_dists_hard[lyr_size] = self.r_prior_dist(size=lyr_size)
    
    def load_hard_r_prior_dist(self, size):
        return self.r_dists_hard[size]
    
    '''
    test
    not actually stable - conformity to Li softmax (normal, no k)
    '''
    def softmax(self, vector):

        # Compute the exponentials of the vector
        exp_vector = np.exp(vector)
        # Compute the softmax values
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    # def stable_softmax(self, vector, k=1):
    #     # Subtract the maximum value from the vector for numerical stability
    #     shift_vector = vector - np.max(vector)
    #     # Compute the exponentials of the shifted vector
    #     exp_vector = np.exp(k * shift_vector)
    #     # Compute the softmax values
    #     softmax_vector = exp_vector / np.sum(exp_vector)
    #     return softmax_vector
    
    def reset_rs_gteq1(self, all_hlyr_sizes, prior_dist):
        n = self.num_layers
        for i in range(1, n + 1):
            self.r[i] = prior_dist(size=all_hlyr_sizes[i - 1])

    # Prints and sends to log file
    def print_and_log(self, *args, **kwargs):
        # Print to the terminal
        print(*args, **kwargs)
        # Print to the file
        exp_log_path = join('models/log',self.exp_log_name)
        if not exists(exp_log_path):
            raise FileNotFoundError(f"Log file {exp_log_path} not found.")
        with open(exp_log_path, "a") as f:
            print(*args, **kwargs, file=f)
    
    def train(self, X, Y, save_checkpoint=None, online_diagnostics=False, plot=False):
        
        printlog = self.print_and_log
        
        '''
        clean up
        '''
        printlog('\n\n')
        printlog(f'priors: {self.priors} init preview:')
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} shape: {self.r[i].shape}')
            printlog(f'r{i} first 3: {self.r[i][:3]}')
            if i > 0:
                # Check if the array is 3D or 5D
                printlog(f'U{i} shape: {self.U[i].shape}')
                if self.U[i].ndim == 3:
                    printlog(f'U{i} first 3x3x3: {self.U[i][:3, :3, :3]}')
                elif self.U[i].ndim == 5:
                    printlog(f'U{i} first 3x3x3x3x3: {self.U[i][:3, :3, :3, :3, :3]}')
                else:
                    printlog(f'U{i} first 3x3: {self.U[i][:3, :3]}')
        if self.classif_method == 'c2':
            printlog(f'Uo shape: {self.U["o"].shape}')
            printlog(f'Uo first 3x3: {self.U["o"][:3, :3]}')
        printlog(f'einsum arg U1: {self.einsum_arg_U1}')
            
        '''
        clean up
        '''
        
        num_imgs = self.num_imgs
        num_tiles = self.num_tiles
        '''
        test: re-shape 3392,864 (num imgs * tiles per image, flattened tile) to 212, 16, 864 (num imgs, tiles per image, flattened tile)
        This will be completed by data.py in the future
        '''
        printlog('test: reshaping X into num imgs, num tiles per image, flattened tile')
        X = X.reshape(num_imgs, num_tiles, -1)

        '''
        test
        '''
        printlog('Train init:')
        printlog('X shape (incl. test reshape):', X.shape)
        printlog('Y shape:', Y.shape)
        
        
        '''
        test
        pre-initiate all distributions
        for speed
        '''
        self.hard_set_prior_dist()
        prior_dist = self.load_hard_prior_dist

        # Else: prior_dist = self.r_prior_dist
        '''
        test
        '''
        
        n = self.num_layers
        all_hlyr_sizes = self.all_hlyr_sizes
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_hlyr_sizes=all_hlyr_sizes)
        
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_all_components = partial(self.update_method_dict[update_method_name], update_method_number)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        rep_cost = self.rep_cost
        
        classif_method = self.classif_method
        classif_cost = self.classif_cost
        
        '''
        test
        '''
        printlog(f'self.kr: {self.kr}')
        printlog(f'self.kU: {self.kU}')
        printlog(f'self.update_method: {self.update_method}')
        printlog(f'update_method_name: {update_method_name}')
        printlog(f'update_method_number: {update_method_number}')
        printlog(f'classif_method: {classif_method}')
        
        '''
        test
        '''
        
        evaluate = partial(self.evaluate, update_method_name=update_method_name, update_method_number=update_method_number, classif_method=classif_method, plot=None)

        if online_diagnostics:
            printlog('\n')
            printlog('Diagnostics on')
            printlog('Epoch: 0')
            epoch = 0
            Jr0 = 0
            Jc0 = 0
            accuracy = 0
            for img in range(num_imgs):
                input = X[img]
                label = Y[img]
                self.r[0] = input
                reset_rs_gteq1(prior_dist=prior_dist)
                update_non_weight_components(label=label)
                Jr0 += rep_cost()
                Jc0 += classif_cost(label)
            accuracy += evaluate(X, Y)
            self.accuracy[epoch] = accuracy
            self.Jr[epoch] = Jr0
            self.Jc[epoch] = Jc0
            printlog(f'Jr: {Jr0}, Jc: {Jc0}, Accuracy: {accuracy}')
        else:
            printlog('\n')
            printlog('Diagnostics: Off')
        
        # Training
        epoch_n = self.epoch_n
        printlog('Training...')
        t_start_train = datetime.now()
        for e in range(epoch_n):
            epoch = e + 1
            printlog(f'Epoch {epoch}')
            t_start_epoch = datetime.now()
            Jre = 0
            Jce = 0
            accuracy = 0
            # Shuffle X, Y
            shuffle_indices = np.random.permutation(num_imgs)
            X_shuff = X[shuffle_indices]
            Y_shuff = Y[shuffle_indices]
            for img in range(num_imgs):
                input = X_shuff[img]
                label = Y_shuff[img]
                self.r[0] = input
                reset_rs(prior_dist=prior_dist)
                update_all_components(label=label)
                if online_diagnostics:
                    Jre += rep_cost()
                    Jce += classif_cost(label)
            if online_diagnostics:
                printlog(f'eval {epoch}')
                accuracy += evaluate(X, Y)
            self.accuracy[epoch] = accuracy
            self.Jr[epoch] = Jre
            self.Jc[epoch] = Jce
            
            # For every 10 epochs, save mid-training diagnostics
            if epoch % 5 == 0:
                # Save mid-training diagnostics
                online_name = self.generate_output_name(self.mod_name, epoch)
                self.save_diagnostics(output_dir='models/', output_name=online_name)
            
            printlog(f'Jr: {Jre}, Jc: {Jce}, Accuracy: {accuracy}')
            t_end_epoch = datetime.now()
            printlog(f'Epoch time: {t_end_epoch - t_start_epoch}.')
            printlog(f'Est. time remaining: {(t_end_epoch - t_start_epoch) * (epoch_n - epoch)}.')
            if epoch == 1:
                printlog(f'Est. tot time: {(t_end_epoch - t_start_epoch) * epoch_n}.')
            
        printlog('Training complete.')
        tot_time = t_end_epoch - t_start_train
        printlog(f'Tot time: {tot_time}.')
        printlog('Saving final model...')
        # Save final model
        final_name = self.generate_output_name(self.mod_name, epoch)
        self.save_model(output_dir='models/', output_name=final_name)
        # Save final diagnostics
        self.save_diagnostics(output_dir='models/', output_name=final_name)
        
        # Final diagnostics
        # Functionize later
        printlog('\n\n')
        printlog(f'Final diagnostics over {epoch} epochs:')
        printlog(f'Ep. 0 Jr: {self.Jr[0]}, Jc: {self.Jc[0]}, Accuracy: {self.accuracy[0]}')
        printlog(f'Ep. {epoch} Jr: {self.Jr[epoch]}, Jc: {self.Jc[epoch]}, Accuracy: {self.accuracy[epoch]}')
        percent_diff_Jr = (self.Jr[0] - self.Jr[epoch]) / self.Jr[0] * 100 if self.Jr[0] != 0 else 0
        percent_diff_Jc = (self.Jc[0] - self.Jc[epoch]) / self.Jc[0] * 100 if self.Jc[0] != 0 else 0
        percent_diff_accuracy = (self.accuracy[0] - self.accuracy[epoch]) / self.accuracy[0] * 100 if self.accuracy[0] != 0 else 0
        printlog(f'Percent diff Jr: {percent_diff_Jr}, Jc: {percent_diff_Jc}, Accuracy: {percent_diff_accuracy}')
        change_per_epoch_Jr = percent_diff_Jr / epoch
        change_per_epoch_Jc = percent_diff_Jc / epoch
        change_per_epoch_accuracy = percent_diff_accuracy / epoch
        printlog(f'Change per epoch Jr: {change_per_epoch_Jr}, Jc: {change_per_epoch_Jc}, Accuracy: {change_per_epoch_accuracy}')
        change_per_min_Jr = percent_diff_Jr / tot_time.total_seconds() * 60
        change_per_min_Jc = percent_diff_Jc / tot_time.total_seconds() * 60
        change_per_min_accuracy = percent_diff_accuracy / tot_time.total_seconds() * 60
        printlog(f'Change per min Jr: {change_per_min_Jr}, Jc: {change_per_min_Jc}, Accuracy: {change_per_min_accuracy}')
        
        # Add a row to models/experiments.csv
        csv_file_path = 'models/experiments.csv'
        csv_columns = ["classif_method", "update_method_name", "kr", "kU", "epochs", "Jr 0", "Jr Final", "Jr % Change", "Jc 0", "Jc Final", "Jc % Change", "Tot Time", "Acc 0", "Acc Final", "Acc % Change"]
        csv_data = [
            classif_method,
            update_method_name,
            self.kr[1],
            self.kU[1],
            epoch,
            self.Jr[0],
            self.Jr[epoch],
            percent_diff_Jr,
            self.Jc[0],
            self.Jc[epoch],
            percent_diff_Jc,
            tot_time,
            self.accuracy[0],
            self.accuracy[epoch],
            percent_diff_accuracy
        ]

        # Ensure the directory exists
        makedirs(dirname(csv_file_path), exist_ok=True)

        # Write to the CSV file
        file_exists = isfile(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_columns)  # Write the header only if the file does not exist
            writer.writerow(csv_data)
        
        
        if plot:
            pass

    def evaluate(self, X, Y, update_method_name, update_method_number, classif_method, plot=None):
        
        '''
        training test
        pre-initiate all distributions
        for speed
        '''

        prior_dist = self.load_hard_prior_dist
        # Else: prior_dist = self.r_prior_dist
        '''
        test
        '''
        reset_rs = partial(self.reset_rs, all_hlyr_sizes=self.all_hlyr_sizes)
        
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        classify = self.classify
        
        n = self.num_layers
        num_imgs = self.num_imgs
        accuracy = 0
        for img in range(num_imgs):
            input = X[img]
            label = Y[img]
            self.r[0] = input
            reset_rs(prior_dist=prior_dist)
            update_non_weight_components(label=label)
            guess = classify(label)
            accuracy += guess
        accuracy /= num_imgs
    
        return accuracy
    
    def predict(self, X, plot=None):

        return None

        
    def update_method_rWniters(self, niters, label, component_updates):
        '''
        Li def: 30
        '''
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        for _ in range(niters):
            r_updates(label)
            for w in range_num_weight_updates:
                # For as many weight sets are there are to update, update them.
                weight_updates[w](label)
        
    def update_method_r_niters_W(self, niters, label, component_updates):
        '''
        Rogers/Brown def: 100
        '''
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        for _ in range(niters):
            r_updates(label)
        for w in range_num_weight_updates:
            # For as many weight sets are there are to update, update them.
            weight_updates[w](label)

    def update_method_r_eq_W(self, stop_criterion, label, component_updates):

        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        num_layers = self.num_layers
        n = num_layers
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, n + 1)]
        diffs = [float('inf')] * n  # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, n + 1)}  # Copy all vectors to avoid reference issues
            r_updates(label)
            
            for i in range(1, n + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change
        
        for w in range_num_weight_updates:
            # For as many weight sets are there are to update, update them.
            weight_updates[w](label) 
            
    def update_method_r_niters(self, niters, label, component_updates):
        '''
        Li def: 30, Rogers/Brown def: 100
        '''
        r_updates = component_updates[0]
        
        for _ in range(niters):
            r_updates(label)

    def update_method_r_eq(self, stop_criterion, label, component_updates):
        '''
        Rogers/Brown def: 0.05
        '''
        
        r_updates = component_updates[0]
        
        num_layers = self.num_layers
        n = num_layers
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, n + 1)]
        diffs = [float('inf')] * n # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, n + 1)}  # Copy all vectors to avoid reference issues
            r_updates(label)
            
            for i in range(1, n + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change
                
    def generate_output_name(self, base_name, epoch):
        # Split the base name at the last underscore
        parts = base_name.rsplit('_', 1)
        # Insert the epoch number before the .pydb extension
        new_name = f"{parts[0]}_{epoch}.pydb"
        return new_name
    
    def save_model(self, output_dir, output_name):
        makedirs(output_dir, exist_ok=True)
        output_path = join(output_dir, output_name)
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
            
    def save_diagnostics(self, output_dir, output_name):
        makedirs(output_dir, exist_ok=True)
        output_name = 'diag.' + output_name
        output_path = join(output_dir, output_name)
        with open(output_path, 'wb') as f:
            pickle.dump({'Jr': self.Jr, 'Jc': self.Jc, 'accuracy':self.accuracy}, f)
            
            
            
class StaticPCC(PredictiveCodingClassifier):

    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        # This is a safeguard for now, as PCC doesn't actually have any init logic but setting attrs.
        # Initialize the base class
        super().__init__()

        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)

        # Component cost funcs: r, U, Uo updates, and cost calculators
        n = self.num_layers
        if n == 1:
            self.r_updates = self.r_updates_n_1
            self.U_updates = self.U_updates_n_1
            self.rep_cost = self.rep_cost_n_1
        elif n == 2:
            self.r_updates = self.r_updates_n_2
            self.U_updates = self.U_updates_n_gt_eq_2
            self.rep_cost = self.rep_cost_n_2
        elif n >= 3:
            self.r_updates = self.r_updates_n_gt_eq_3
            self.U_updates = self.U_updates_n_gt_eq_2
            self.rep_cost = self.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
        # Classification now
        self.classif_cost = self.classif_cost_dict[self.classif_method]
        self.classify = self.classif_guess_dict[self.classif_method]
        
        # Components together
        self.component_updates = [self.r_updates, self.U_updates]
        if self.classif_method == 'c2':
            self.component_updates.append(self.Uo_update)

        
        self.update_method_dict = {'rW_niters': partial(self.update_method_rWniters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters_W, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq_W, component_updates=self.component_updates)}
        
        self.update_method_no_weight_dict = {'rW_niters': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq, component_updates=self.component_updates)}
        
        self.rn_topdown_upd_dict = {'c1': self.rn_topdown_upd_c1,
                                    'c2': self.rn_topdown_upd_c2,
                                    None: self.rn_topdown_upd_None}
        
        self.classif_cost_dict = {'c1': self.classif_cost_c1,
                                'c2': self.classif_cost_c2,
                                None: self.classif_cost_None}
        
        self.classif_guess_dict = {'c1': self.classif_guess_c1,
                                'c2': self.classif_guess_c2,
                                None: self.classif_guess_None}
        
    def validate_attributes(self):
        pass
    
class RecurrentPCC(PredictiveCodingClassifier):

    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        # This is a safeguard for now, as PCC doesn't actually have any init logic but setting attrs.
        # Initialize the base class
        super().__init__()

        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)
        
        # General method, grabs r,U (Uo, V when applicable) update funcs and cost funcs
        self.set_component_update_funcs(num_layers=self.num_layers)
        
        # Component updates: r, U, Uo, and cost calculato
        n = self.num_layers
        if n == 1:
            self.r_updates = self.r_updates_n_1
            self.U_updates = self.U_updates_n_1
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_1
        elif n == 2:
            self.r_updates = self.r_updates_n_2
            self.U_updates = self.U_updates_n_gt_eq_2
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_2
        elif n >= 3:
            self.r_updates = self.r_updates_n_gt_eq_3
            self.U_updates = self.U_updates_n_gt_eq_2
            self.V_updates = self.V_updates
            self.rep_cost = self.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
        self.component_updates = [self.r_updates, self.U_updates]
        classif_method = self.classif_method
        if classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
        # Add Vs
        self.component_updates.append(self.V_updates)