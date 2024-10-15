from cost_functions import StaticCostFunction
import numpy as np
from functools import partial

from datetime import datetime
import pickle
from os import makedirs
from os.path import join, exists, dirname, isfile
# May have to remove if cluster
import matplotlib.pyplot as plt

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
        
        self.softmax_dict = {'normal': self.softmax,
                            'stable': self.stable_softmax}
        
        '''
        shell class for sPCC and rPCC subclasses
        inheritance: some general methods, and all attributes
        '''
        
    def set_model_attributes(self, params):
        '''
        Set model attributes from a dictionary.
        This will set a bunch of external attributes too,
        which will serve no other purpose than to recount the last experiment run on the model. e.g. name, train, notes.
        '''
        for key, value in params.items():
            setattr(self, key, value)
            
    def set_Idiff_tdot_dims(self, input_size):
        # All architectures
        # Input dims, all
        ndims_input = len(input_size)
        range_ndims_input = range(ndims_input)
        # L1 Bottom-up error dims for Idiff^2 calculation in rep_cost()
        self.Idiff_tdot_dims = list(range_ndims_input)
        
    def set_L1diff_tdot_dims(self, L1_size):
        # All architectures
        # L1 (r1) dims, all
        ndims_input = len(L1_size)
        range_ndims_input = range(ndims_input)
        # L1's Top-down error dims for L1diff^2 calculation in rep_cost()
        self.L1diff_tdot_dims = list(range_ndims_input)
            
    def set_U1_einsum_arg(self):
        # Only applicable to 'expand_first_lyr'
        # 5d U1 case, 3d r1 case (tiled nonflat)
        if self.tiled_input and not self.flat_input:
            self.U1_einsum_arg = 'ijklm,ijm-> ijkl'
        # If tiled flat, untiled flat, or untiled nonflat
        # 3d U1 case, 2d r1 case
        else:
            self.U1_einsum_arg = 'ijk,ik->ij'
            
    def set_U1T_dims(self, U1_size):
        # Applicable to all architectures, except e1L_Li
        # 'Transpose' (permute) dims
        if not self.architecture == 'expand_first_lyr_Li':
            ndims_U1 = len(U1_size)
            range_ndims_U1 = range(ndims_U1)
            last_dim_id_U1 = range_ndims_U1[-1]
            nonlast_dim_ids_U1 = range_ndims_U1[:-1]
            transpose_dims_U1 = tuple([last_dim_id_U1] + list(nonlast_dim_ids_U1))
            self.U1T_dims = transpose_dims_U1
        # e1L_Li
        else:
            # Li permutation
            self.U1T_dims = (0, 2, 1)

    def set_U1T_tdot_dims(self, U1_size):
        # Only affects 'flat hidden layers' architecture
        # For U1T Idiff calculation
        # Tensordot dims
        # U1T, last n-1
        ndims_U1 = len(U1_size)
        range_ndims_U1 = range(ndims_U1)
        nonfirst_dim_ids_U1 = range_ndims_U1[1:]
        self.U1T_tdot_dims = list(nonfirst_dim_ids_U1)
            
    def set_U1T_einsum_arg(self):
        # Only affects 'expand first lyr' architecture
        # For U1T Idiff calculation
        # 5d U1 case, 3d r1 case (tiled nonflat)
        if self.tiled_input and not self.flat_input:
            self.U1T_einsum_arg = 'ijklm,jklm->jki'
        # Otherwise
        else:
            self.U1T_einsum_arg = 'ijk,jk->ji'
            
    def set_U2T_dims(self, U2_size):
        # Applicable to all architectures
        # Transpose dims
        # Same protocol as U1T.
        ndims_U2 = len(U2_size)
        range_ndims_U2 = range(ndims_U2)
        last_dim_id_U2 = range_ndims_U2[-1]
        nonlast_dim_ids_U2 = range_ndims_U2[:-1]
        transpose_dims_U2 = tuple([last_dim_id_U2] + list(nonlast_dim_ids_U2))
        self.U2T_dims = transpose_dims_U2
        
    def set_U2T_einsum_arg(self):
        # Only affects 'expand first lyr' architecture
        # For U2T L1diff calculation
        # 5d U1 case, 3d r1 case (tiled nonflat)
        if self.tiled_input and not self.flat_input:
            self.U2T_einsum_arg = 'ijkl,jkl->i'
        # Otherwise
        else:
            self.U2T_einsum_arg = 'ijk,jk->i'
            
    def set_Idiff_einsum_arg(self):
        # Only affects 'expand first lyr' and 'flat_hidden_lyrs' architectures
        # For Idiff ('outer') r1 calculation
        # If 'expand first lyr'
        if self.architecture == 'expand_first_lyr':
            # 5d U1 case, 3d r1 case (tiled nonflat)
            if self.tiled_input and not self.flat_input:
                self.Idiff_einsum_arg = 'ijkl,ijm->ijklm'
            # Otherwise
            else:
                self.Idiff_einsum_arg = 'ij,ik->ijk'
        # Flat
        elif self.architecture == 'flat_hidden_lyrs':
            # 5d U1 case, 3d r1 case (tiled nonflat)
            if self.tiled_input and not self.flat_input:
                self.Idiff_einsum_arg = 'ijkl,m->ijklm'
            # Otherwise
            else:
                self.Idiff_einsum_arg = 'ij,k->ijk'
            
    def set_L1diff_einsum_arg(self):
        # Only affects 'expand first lyr'  architecture
        # For L1diff ('outer') r2 calculation
        # 5d U1 case, 3d r1 case (tiled nonflat)
        if self.tiled_input and not self.flat_input:
            self.Idiff_einsum_arg = 'ijk,l->ijkl'
        # Otherwise
        else:
            self.Idiff_einsum_arg = 'ij,k->ijk'
    
    def config_from_attributes(self):
        '''
        Set up the model from the attributes.
        '''
        # Not made
        #  self.validate_attributes()
        
        # Transforms and priors
        self.f = self.act_fxn_dict[self.activ_func]
        self.g = self.prior_cost_dict[self.priors]
        self.h = self.prior_cost_dict[self.priors]
        
        # Lr denominators (Brown method or Li method)
        if self.lr_denominators == "Li_denominators":
            # Prior terms
            self.lr_prior_denominator = 1
            # rn top-down update term
            self.lr_rn_td_denominator = self.ssq[self.num_layers]
        elif self.lr_denominators == "Brown_denominators":
            self.lr_prior_denominator = 2
            # rn top-down update term
            self.lr_rn_td_denominator = 2

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
        
        # Params needed
        tiled_input = self.tiled_input
        flat_input = self.flat_input
        architecture = self.architecture
        input_shape = self.input_shape
        num_layers = self.num_layers
        n = num_layers

        # Inits on representations and weights
        self.r = {}
        self.U = {}
        # r0
        self.r[0] = np.zeros(input_shape)
        for i in range(1, n + 1):
            # Layer 1
            if i == 1:
                # r1
                if architecture == 'flat_hidden_lyrs':
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
                        print('r', self.r)
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
        
        # Initiate U1 and U2-based operations dims (dimentions flex based on input and architecture)
        # If the rep cost is being calculated
        self.set_Idiff_tdot_dims(input_shape)
        self.set_L1diff_tdot_dims(self.r[1].shape)
        # Used in U "dot" r at higher dims
        self.set_U1_einsum_arg()
        # Transposes
        self.set_U1T_dims(U1_size)
        self.set_U2T_dims(U2_size)
        # U1-2T stuff
        self.set_U1T_tdot_dims(U1_size)
        self.set_U1T_einsum_arg()
        self.set_U2T_einsum_arg()
        # U-only stuff
        self.set_Idiff_einsum_arg()
        self.set_L1diff_einsum_arg()
        
        # All layer sizes for priors
        self.all_lyr_sizes = self.hidden_lyr_sizes.copy()
        self.all_lyr_sizes.append(self.output_lyr_size)
        if architecture == 'expand_first_lyr_Li' or architecture == 'expand_first_lyr':
            self.all_lyr_sizes[0] = self.r[1].shape
            
        # Set softmax
        self.softmax_func = partial(self.set_softmax_func, softmax_type=self.softmax_type, k=self.softmax_k)
        
        # Initiate Jr, Jc, and accuracy (diagnostics) for storage, print, plot
        epoch_n = self.epoch_n
        self.Jr = [0] * (epoch_n + 1)
        self.Jc = [0] * (epoch_n + 1)
        self.accuracy = [0] * (epoch_n + 1)
        
        # Later: by layer
        # self.Jr = {i: [0] * (epoch_n + 1) for i in range(n)}
        # self.Jc = {i: [0] * (epoch_n + 1) for i in range(n)}
        
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
            lyr_size = self.all_lyr_sizes[i - 1]
            self.r_dists_hard[lyr_size] = self.r_prior_dist(size=lyr_size)
    
    def load_hard_r_prior_dist(self, size):
        return self.r_dists_hard[size]
    
    def softmax(self, vector, k=1):

        # Compute the exponentials of the vector
        exp_vector = np.exp(k * vector)
        # Compute the softmax values
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def stable_softmax(self, vector, k=1):
        # Subtract the maximum value from the vector for numerical stability
        shift_vector = vector - np.max(vector)
        # Compute the exponentials of the shifted vector
        exp_vector = np.exp(k * shift_vector)
        # Compute the softmax values
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def set_softmax_func(self, softmax_type, vector, k):
        return self.softmax_dict[softmax_type](vector=vector, k=k)
    
    def reset_rs_gteq1(self, all_lyr_sizes, prior_dist):
        n = self.num_layers
        for i in range(1, n + 1):
            self.r[i] = prior_dist(size=all_lyr_sizes[i - 1])

    # Prints and sends to log file
    def print_and_log(self, *args, **kwargs):
        # Print to the terminal
        print(*args, **kwargs)
        # Print to the file
        exp_log_path = join('log',self.exp_log_name)
        if not exists(exp_log_path):
            raise FileNotFoundError(f"Log file {exp_log_path} not found.")
        with open(exp_log_path, "a") as f:
            print(*args, **kwargs, file=f)
    
    def train(self, X, Y, save_checkpoint=None, load_checkpoint=None, plot=False):
        
        printlog = self.print_and_log
        
        '''
        clean up
        '''
        printlog('\n\n')
        printlog('shapes')
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} shape: {self.r[i].shape}')
            if i > 0:
                # Check if the array is 3D or 5D
                printlog(f'U{i} shape: {self.U[i].shape}')
        if self.classif_method == 'c2':
            printlog(f'Uo shape: {self.U["o"].shape}')
        printlog('\n')
            
        printlog(f'priors: {self.priors} init preview:')
        for i in range(0, self.num_layers + 1):
            printlog(f'r{i} first 3: {self.r[i][:3]}')
            if i > 0:
                # Check if the array is 3D or 5D
                if self.U[i].ndim == 3:
                    printlog(f'U{i} first 3x3x3: {self.U[i][:3, :3, :3]}')
                elif self.U[i].ndim == 5:
                    printlog(f'U{i} first 3x3x3x3x3: {self.U[i][:3, :3, :3, :3, :3]}')
                else:
                    printlog(f'U{i} first 3x3: {self.U[i][:3, :3]}')
        if self.classif_method == 'c2':
            printlog(f'Uo first 3x3: {self.U["o"][:3, :3]}')
        printlog('\n')
            
        '''
        clean up
        '''
        
        epoch_n = self.epoch_n
        num_imgs = self.num_imgs
        num_tiles = self.num_tiles
        '''
        test: re-shape 3392,864 (num imgs * tiles per image, flattened tile) to 212, 16, 864 (num imgs, tiles per image, flattened tile)
        This will be completed by data.py in the future, by God's grace.
        '''
        printlog('original X shape:', X.shape)
        X = X.reshape(num_imgs, num_tiles, -1)
        printlog('test: reshaping X into num imgs, num tiles per image, flattened tile')

        '''
        test
        '''
        printlog('Train init:')
        printlog('X shape:', X.shape)
        printlog('Y shape:', Y.shape)
        
        
        '''
        pre-initiate all distributions
        for speed
        '''
        self.hard_set_r_prior_dist()
        prior_dist = self.load_hard_r_prior_dist
        # Else: prior_dist = self.r_prior_dist
        '''
        '''

        # Methods
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_all_components = partial(self.update_method_dict[update_method_name], update_method_number)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        rep_cost = self.rep_cost
        
        classif_method = self.classif_method
        classif_cost = self.classif_cost
        
        evaluate = partial(self.evaluate, update_method_name=update_method_name, update_method_number=update_method_number, classif_method=classif_method, plot=None)
        
        # Checkpointing
        if 'save_every' in save_checkpoint:
            # If N
            checkpoint = save_checkpoint['save_every'] 
        elif 'fraction' in save_checkpoint:
            fraction_value = save_checkpoint['fraction']
            # If frac tuple
            if isinstance(fraction_value, tuple):
                numerator, denominator = fraction_value
                checkpoint = (numerator/denominator) * epoch_n
            # If it's a decimal
            else:
                checkpoint = fraction_value * epoch_n
            # Round up to the nearest whole number
            checkpoint = np.ceil(checkpoint)
        else:
            checkpoint = None  # Default case if neither key is found
        printlog(f'Checkpoint method: {save_checkpoint}', '\n', 'saving every', checkpoint)
        
        # If loading
        if load_checkpoint is not None:
            printlog(f'Load checkpoint method: {load_checkpoint}')
            # # Then add the number of epochs it already has to the starting '0'
            # load_name = self.load_name 
            # # take off .pydb
            # load_name_no_pydb = load_name.rsplit('.', 1)[0]
            # # isolate epoch number
            # load_epoch = int(load_name_no_pydb.rsplit('_', 1)[1])
            # epoch = load_epoch
            start_epoch = self.load_epoch
            printlog('starting epoch is', epoch)
        else:
            start_epoch = 0

        # Epoch '0' evaluation (pre-training, or if checkpoint has been loaded, pre-additional-training)
        Jr0 = 0
        Jc0 = 0
        accuracy = 0
        printlog('\n')
        printlog(f'Epoch: {start_epoch}')
        for img in range(num_imgs):
            input = X[img]
            label = Y[img]
            self.r[0] = input
            reset_rs_gteq1(prior_dist=prior_dist)
            update_non_weight_components(label=label)
            Jr0 += rep_cost()
            Jc0 += classif_cost(label)
        accuracy += evaluate(X, Y)
        printlog(f'Jr: {Jr0}, Jc: {Jc0}, Accuracy: {accuracy}')
        self.Jr[start_epoch] = Jr0
        self.Jc[start_epoch] = Jc0
        self.accuracy[start_epoch] = accuracy
        
        # Training
        printlog('Training...')
        t_start_train = datetime.now()
        for e in range(epoch_n):
            epoch = e + 1 + start_epoch
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
                reset_rs_gteq1(prior_dist=prior_dist)
                update_all_components(label=label)
                Jre += rep_cost()
                Jce += classif_cost(label)
            printlog(f'eval {epoch}')
            accuracy += evaluate(X, Y)
            self.Jr[epoch] = Jre
            self.Jc[epoch] = Jce
            self.accuracy[epoch] = accuracy
            
            '''
            hard-coded in here for KB investigation'''
            
            # For every 10 epochs, save mid-training diagnostics
            if epoch % 10 == 0:
                # Save mid-training diagnostics
                online_name = self.generate_output_name(self.mod_name, epoch)
                self.save_diagnostics(output_dir='results/diagnostics/', output_name=online_name)
                if plot: 
                    self.plot(input_dir='results/diagnostics/', input_name=online_name)
            ''''''
            
            # Checkpointing
            if checkpoint is not None and epoch % checkpoint == 0:
                # Save checkpoint model
                checkpoint_name = self.generate_output_name(self.mod_name, epoch)
                self.save_model(output_dir='models/checkpoints/', output_name=checkpoint_name)
                checkpoint_path = join('models/checkpoints/', checkpoint_name)
                printlog(f'Checkpoint model saved at epoch {epoch}:\n{checkpoint_path}')
            
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
        self.save_diagnostics(output_dir='results/diagnostics/', output_name=final_name)
        
        # Plot
        if plot:
            self.plot(input_dir='results/diagnostics/', input_name=final_name)
        
        # Final diagnostics terminal readout
        # Functionize later
        printlog(f'Final diagnostics over {epoch} epochs:')
        printlog(f'Ep. 0 Jr: {self.Jr[0]}, Jc: {self.Jc[0]}, Accuracy: {self.accuracy[0]}')
        printlog(f'Ep. {epoch} Jr: {self.Jr[epoch]}, Jc: {self.Jc[epoch]}, Accuracy: {self.accuracy[epoch]}')

        # Corrected percent difference calculation
        epsilon = 1e-6
        percent_diff_Jr = ((self.Jr[0] - self.Jr[epoch]) / (self.Jr[0] + epsilon)) * 100
        percent_diff_Jc = ((self.Jc[0] - self.Jc[epoch]) / (self.Jc[0] + epsilon)) * 100
        percent_diff_accuracy = ((self.accuracy[0] - self.accuracy[epoch]) / (self.accuracy[0] + epsilon)) * 100
        
        printlog(f'Percent diff Jr: {percent_diff_Jr}%, Jc: {percent_diff_Jc}%, Accuracy: {percent_diff_accuracy}%')
        
        change_per_epoch_Jr = percent_diff_Jr / epoch
        change_per_epoch_Jc = percent_diff_Jc / epoch
        change_per_epoch_accuracy = percent_diff_accuracy / epoch
        
        printlog(f'Change per epoch Jr: {change_per_epoch_Jr}%, Jc: {change_per_epoch_Jc}%, Accuracy: {change_per_epoch_accuracy}%')
        
        change_per_min_Jr = percent_diff_Jr / tot_time.total_seconds() * 60
        change_per_min_Jc = percent_diff_Jc / tot_time.total_seconds() * 60
        change_per_min_accuracy = percent_diff_accuracy / tot_time.total_seconds() * 60
        
        printlog(f'Change per min Jr: {change_per_min_Jr}%, Jc: {change_per_min_Jc}%, Accuracy: {change_per_min_accuracy}%')

    def evaluate(self, X, Y, update_method_name, update_method_number, classif_method, plot=None):
        
        '''
        training test
        pre-initiate all distributions
        for speed
        '''
        prior_dist = self.load_hard_r_prior_dist
        # Else: prior_dist = self.r_prior_dist
        '''
        test
        '''
        
        reset_rs_gteq1 = partial(self.reset_rs_gteq1, all_lyr_sizes=self.all_lyr_sizes)
        
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        classify = self.classify
        num_imgs = self.num_imgs
        accuracy = 0
        for img in range(num_imgs):
            input = X[img]
            label = Y[img]
            self.r[0] = input
            reset_rs_gteq1(prior_dist=prior_dist)
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
            
    def plot(self, input_dir, input_name):
        
        printlog = self.print_and_log
        
        diags_name = 'diag.' + input_name
        diags_path = join(input_dir, diags_name)
        
        # Load the model
        with open(diags_path, 'rb') as file:
            diags = pickle.load(file)
        
        printlog(f'\nplot:\nloaded diag file:\n{diags_path}')
        
        # Split the filename by _ and take the last, then split by . and take the first. This is epoch_n
        epoch_n = int(diags_name.split('_')[-1].split('.')[0])
        
        # clip diags at epoch_n
        diags['Jr'] = diags['Jr'][:epoch_n]
        diags['Jc'] = diags['Jc'][:epoch_n]
        diags['accuracy'] = diags['accuracy'][:epoch_n]
            
        epochs = range(len(diags['Jr']))
            
        # Turn each accuracy into an actual percent
        diags['accuracy'] = [acc * 100 for acc in diags['accuracy']]

        # Calculate percent changes
        epsilon = 1e-6
        percent_change_Jr = ((diags['Jr'][-1] - diags['Jr'][0]) / (diags['Jr'][0] + epsilon)) * 100
        percent_change_Jc = ((diags['Jc'][-1] - diags['Jc'][0]) / (diags['Jc'][0] + epsilon)) * 100
        percent_change_accuracy = ((diags['accuracy'][-1] - diags['accuracy'][0]) / (diags['accuracy'][0] + epsilon)) * 100

        fig, ax1 = plt.subplots()

        # Plot Jr
        ax1.plot(epochs, diags['Jr'], 'y-', label='Jr')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Jr', color='y')
        ax1.tick_params(axis='y', labelcolor='y')
        ax1.set_ylim(0, max(diags['Jr']) * 1.05)

        # Create a second y-axis for Jc
        ax2 = ax1.twinx()
        ax2.plot(epochs, diags['Jc'], 'b-', label='Jc')
        ax2.set_ylabel('Jc', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(0, max(diags['Jc']) * 1.05)

        # Create a third y-axis for accuracy
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(epochs, diags['accuracy'], 'r-', label='Accuracy', linewidth=0.5)
        ax3.set_ylabel('Accuracy (%)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_ylim(min(diags['accuracy']) * 1.05, max(diags['accuracy']) * 1.05)

        # Add legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='center left', bbox_to_anchor=(-.3, .9))

        # Add percent change text
        plt.text(0.1, 0.95, f'Jr % Change: {percent_change_Jr:.2f}%', transform=ax1.transAxes, color='y')
        plt.text(0.1, 0.90, f'Jc % Change: {percent_change_Jc:.2f}%', transform=ax1.transAxes, color='b')
        plt.text(0.1, 0.85, f'Accuracy % Change: {percent_change_accuracy:.2f}%', transform=ax1.transAxes, color='r')
        
        plt.title('Training Diagnostics\n' + diags_name + '\n')
        
        results_folder = 'results/plots'
        results_path = join(results_folder, diags_name)
        fig.savefig(results_path + '.png', dpi=300, bbox_inches='tight')
        
        printlog(f'Saved plot:\n',results_path,'\n')
            
        
class StaticPCC(PredictiveCodingClassifier):

    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        # This is a safeguard for now, as PCC doesn't actually have any init logic but setting attrs.
        # Initialize the base class
        super().__init__()

        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)
        
        # Cost functions
        static_cost_func_class = StaticCostFunction(self)
        
        # Component cost funcs: r, U, Uo updates, and cost calculators
        n = self.num_layers
        if n == 1:
            self.r_updates = static_cost_func_class.r_updates_n_1
            self.U_updates = static_cost_func_class.U_updates_n_1
            self.rep_cost = static_cost_func_class.rep_cost_n_1
        elif n == 2:
            self.r_updates = static_cost_func_class.r_updates_n_2
            self.U_updates = static_cost_func_class.U_updates_n_gt_eq_2
            self.rep_cost = static_cost_func_class.rep_cost_n_2
        elif n >= 3:
            self.r_updates = static_cost_func_class.r_updates_n_gt_eq_3
            self.U_updates = static_cost_func_class.U_updates_n_gt_eq_2
            self.rep_cost = static_cost_func_class.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
        # Components together
        self.component_updates = [self.r_updates, self.U_updates]
        if self.classif_method == 'c2':
            self.component_updates.append(self.Uo_update)

        # Dictionaries for methods from base class and static cost function class
        self.update_method_dict = {'rW_niters': partial(self.update_method_rWniters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters_W, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq_W, component_updates=self.component_updates)}
        
        self.update_method_no_weight_dict = {'rW_niters': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq, component_updates=self.component_updates)}
        
        self.classif_cost_dict = {'c1': static_cost_func_class.classif_cost_c1,
                                'c2': static_cost_func_class.classif_cost_c2,
                                None: static_cost_func_class.classif_cost_None}
        
        self.classif_guess_dict = {'c1': static_cost_func_class.classif_guess_c1,
                                'c2': static_cost_func_class.classif_guess_c2,
                                None: static_cost_func_class.classif_guess_None}
        
        # Classification now
        self.classif_cost = self.classif_cost_dict[self.classif_method]
        self.classify = self.classif_guess_dict[self.classif_method]
    
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