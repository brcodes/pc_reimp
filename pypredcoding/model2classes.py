from parameters import constant_lr, step_decay_lr, polynomial_decay_lr
import numpy as np
from functools import partial

from datetime import datetime
import pickle
from os import path, makedirs
import io
import contextlib


class PredictiveCodingClassifier:

    def __init__(self):
        
        # Choices for transformation functions, priors
        self.act_fxn_dict = {'linear': self.linear_transform,
                                'tanh': self.tanh_transform}
        self.prior_cost_dict = {'gaussian': self.gaussian_prior_costs, 
                                'kurtotic': self.kurtotic_prior_costs}
        self.prior_dist_dict = {'gaussian': partial(np.random.normal, loc=0, scale=1),
                                'kurtotic': partial(np.random.laplace, loc=0.0, scale=0.5)}
        
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
        
        # Transforms and priors
        self.f = self.act_fxn_dict[self.activ_func]
        self.g = self.prior_cost_dict[self.priors]
        self.h = self.prior_cost_dict[self.priors]
        
        self.r = {}
        self.U = {}
        
        # Initiate rs, Us (Vs in recurrent subclass)
        # r0 is going to be different
        input_size = self.input_size
        num_layers = self.num_layers
        n = num_layers
        self.r[0] = np.zeros(input_size)
        print(f'r0 and input shape: {self.r[0].shape}')
        for i in range(1, n + 1):
            if i == n:
                self.r[i] = self.prior_dist(size=(self.output_lyr_size))
            else:
                self.r[i] = self.prior_dist(size=(self.hidden_lyr_sizes[i - 1]))
            print(f'r{i} shape: {self.r[i].shape}')
            print(f'r{i} first 3: {self.r[i][:3]}')
        
        # Initiate Us
        # U1 is going to be a little bit different
        U1_size = tuple(list(input_size) + list(self.r[1].shape))
        self.U[1] = self.prior_dist(size=U1_size)
        # U2 through Un
        for i in range(2, n + 1):
            Ui_size = (self.r[i-1].shape[0], self.r[i].shape[0])
            self.U[i] = self.prior_dist(size=Ui_size)
            print(f'U{i} shape: {self.U[i].shape}')
            print(f'U{i} first 3x3: {self.U[i][:3, :3]}')
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.output_lyr_size)
            self.U['o'] = self.prior_dist(size=Uo_size)
            print(f'Uo shape: {self.U["o"].shape}')
            
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
        
        print(f'einsum arg U1: {self.einsum_arg_U1}')
        
        # Hidden layer sizes for priors
        self.all_hlyr_sizes = self.hidden_lyr_sizes.copy()
        self.all_hlyr_sizes.append(self.output_lyr_size)
        
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
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior.
        """
        g_or_h = alph_or_lam * np.square(r_or_U).sum()
        gprime_or_hprime = 2 * alph_or_lam * r_or_U
        return (g_or_h, gprime_or_hprime)

    # def kurtotic_prior_costs(self, r_or_U=None, alph_or_lam=None):
    #     """
    #     Takes an argument pair of either r & alpha, or U & lambda, and returns
    #     a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
    #     """
        
    #     g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
    #     gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    #     return (g_or_h, gprime_or_hprime)
    
    '''
    test
    '''
    def kurtotic_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
        """
        try:
            # Set NumPy to raise exceptions on overflow and invalid operations
            np.seterr(over='raise', invalid='raise')
            
            g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
            gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
            
            # Reset NumPy error handling to default
            np.seterr(over='warn', invalid='warn')
            
            return (g_or_h, gprime_or_hprime)
        
        except FloatingPointError as e:
            print(f"FloatingPointError: {e}")
            
            if r_or_U is not None:
                if r_or_U.ndim == 1:  # r_or_U is a vector
                    print("First five elements of r:", r_or_U[:5])
                elif r_or_U.ndim == 2:  # r_or_U is a matrix
                    print("First 5x5 elements of U:\n", r_or_U[:5, :5])
            
            return None

    def prior_dist(self, size):
        return self.prior_dist_dict[self.priors](size=size)
    
    def hard_set_prior_dist(self):
        '''
        so a new one isn't made every tiem'''
        
        self.r_dists_hard = {}
        n = self.num_layers
        for i in range (1, n + 1):
            lyr_size = self.all_hlyr_sizes[i - 1]
            self.r_dists_hard[lyr_size] = self.prior_dist(size=lyr_size)
    
    def load_hard_prior_dist(self, size):
        return self.r_dists_hard[size]
    
    def stable_softmax(self, vector, k=1):
        # Subtract the maximum value from the vector for numerical stability
        shift_vector = vector - np.max(vector)
        # Compute the exponentials of the shifted vector
        exp_vector = np.exp(k * shift_vector)
        # Compute the softmax values
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def reset_rs(self, all_hlyr_sizes, prior_dist):
        n = self.num_layers
        for i in range(1, n + 1):
            self.r[i] = prior_dist(size=all_hlyr_sizes[i - 1])
            
    def initiate_logging(self):
        log_dir = 'models/log'
        makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        log_file_path = path.join(log_dir, f'mod_{timestamp}.txt')
        log_buffer = io.StringIO()
        return log_buffer, log_file_path
    
    def train(self, X, Y, save_checkpoint=None, online_diagnostics=False, plot=False):
        
        # log_buffer, log_file_path = self.initiate_logging()
        
        # with contextlib.redirect_stdout(log_buffer):
        
        num_imgs = self.num_imgs
        num_tiles = self.num_tiles
        '''
        test: re-shape 3392,864 (num imgs * tiles per image, flattened tile) to 212, 16, 864 (num imgs, tiles per image, flattened tile)
        This will be completed by data.py in the future
        '''
        print('test: reshaping X into num imgs, num tiles per image, flattened tile')
        X = X.reshape(num_imgs, num_tiles, -1)

        '''
        test
        '''
        print('Train init:')
        print('X shape (incl. test reshape):', X.shape)
        print('Y shape:', Y.shape)
        
        
        '''
        test
        pre-initiate all distributions
        for speed
        '''
        self.hard_set_prior_dist()
        prior_dist = self.load_hard_prior_dist

        # Else: prior_dist = self.prior_dist
        '''
        test
        '''
        
        n = self.num_layers
        all_hlyr_sizes = self.all_hlyr_sizes
        reset_rs = partial(self.reset_rs, all_hlyr_sizes=all_hlyr_sizes)
        
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_all_components = partial(self.update_method_dict[update_method_name], update_method_number)
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        rep_cost = self.rep_cost
        
        classif_method = self.classif_method
        classif_cost = self.classif_cost_dict[classif_method]
        
        evaluate = partial(self.evaluate, update_method_name=update_method_name, update_method_number=update_method_number, classif_method=classif_method, plot=None)
    
        if online_diagnostics:
            print('Diagnostics on')
            print('Epoch: 0')
            epoch = 0
            Jr0 = 0
            Jc0 = 0
            accuracy = 0
            for img in range(num_imgs):
                input = X[img]
                label = Y[img]
                self.r[0] = input
                reset_rs(prior_dist=prior_dist)
                update_non_weight_components(label=label)
                Jr0 += rep_cost()
                Jc0 += classif_cost(label)
            accuracy += evaluate(X, Y)
            self.accuracy[epoch] = accuracy
            self.Jr[epoch] = Jr0
            self.Jc[epoch] = Jc0
            print(f'Jr: {Jr0}, Jc: {Jc0}, Accuracy: {accuracy}')
        else:
            print('Diagnostics: Off')
        
        # Training
        epoch_n = self.epoch_n
        print('Training...')
        t_start_train = datetime.now()
        for e in range(epoch_n):
            epoch = e + 1
            print(f'Epoch {epoch}')
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
                print(f'eval {epoch}')
                accuracy += evaluate(X, Y)
            self.accuracy[epoch] = accuracy
            self.Jr[epoch] = Jre
            self.Jc[epoch] = Jce
            print(f'Jr: {Jre}, Jc: {Jce}, Accuracy: {accuracy}')
            t_end_epoch = datetime.now()
            print(f'Epoch time: {t_end_epoch - t_start_epoch}.')
            print(f'Est. time remaining: {(t_end_epoch - t_start_epoch) * (epoch_n - epoch)}.')
            if epoch == 1:
                print(f'Est. tot time: {(t_end_epoch - t_start_epoch) * epoch_n}.')
            
        print('Training complete.')
        print(f'Tot time: {t_end_epoch - t_start_train}.')
        print('Saving final model...')
        # Save final model
        final_name = self.generate_output_name(self.mod_name, epoch)
        self.save_model(output_dir='models/', output_name=final_name)
        
        if plot:
            pass
            
        # # Write the captured output to the log file
        # with open(log_file_path, 'w') as log_file:
        #     log_file.write(log_buffer.getvalue())

        # log_buffer.close()

    def evaluate(self, X, Y, update_method_name, update_method_number, classif_method, plot=None):
        
        '''
        training test
        pre-initiate all distributions
        for speed
        '''

        prior_dist = self.load_hard_prior_dist
        # Else: prior_dist = self.prior_dist
        '''
        test
        '''
        reset_rs = partial(self.reset_rs, all_hlyr_sizes=self.all_hlyr_sizes)
        
        update_non_weight_components = partial(self.update_method_no_weight_dict[update_method_name], update_method_number)
        
        guess_func = self.classif_guess_dict[classif_method]
        
        n = self.num_layers
        num_imgs = self.num_imgs
        accuracy = 0
        for img in range(num_imgs):
            input = X[img]
            label = Y[img]
            self.r[0] = input
            reset_rs(prior_dist=prior_dist)
            update_non_weight_components(label=label)
            guess = guess_func(label)
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
        output_path = path.join(output_dir, output_name)
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
            
            
            
class StaticPCC(PredictiveCodingClassifier):

    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        # This is a safeguard for now, as PCC doesn't actually have any init logic but setting attrs.
        # Initialize the base class
        super().__init__()

        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)
        
        '''
        test
        '''
        # Component updates: r, U, Uo, and cost calculato
        num_layers = self.num_layers
        n = num_layers
        if n == 1:
            self.r_updates = self.r_updates_n_1_no_transform
            self.U_updates = self.U_updates_n_1_no_transform
            self.rep_cost = self.rep_cost_n_1_no_transform
        elif n == 2:
            self.r_updates = self.r_updates_n_2_no_transform
            self.U_updates = self.U_updates_n_gt_eq_2_no_transform
            self.rep_cost = self.rep_cost_n_2_no_transform
        elif n >= 3:
            self.r_updates = self.r_updates_n_gt_eq_3_no_transform
            self.U_updates = self.U_updates_n_gt_eq_2_no_transform
            self.rep_cost = self.rep_cost_n_gt_eq_3_no_transform
        else:
            raise ValueError("Number of layers must be at least 1.")
        self.component_updates = [self.r_updates, self.U_updates]
        classif_method = self.classif_method
        if classif_method == 'c2':
            self.component_updates.append(self.Uo_update_no_transform)
        '''
        test
        '''
        
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
        
    def rep_cost_n_1(self):
        pass
    
    def rep_cost_n_2(self):
        pass
    
    def rep_cost_n_gt_eq_3(self):
        pass
    
    def rep_cost_n_1_no_transform(self):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        r_0 = self.r[0]
        r_1 = self.r[1]
        U_1 = self.U[1]
        ssq_1 = self.ssq[1]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        
        # 1st layer axes necessary for dot product (2D or 4D)
        bu_tdot_dims = self.bu_error_tdot_dims
        
        # Bottom up only
        bu_v = r_0 - self.f(U1_tdot_r1)[0]
        bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_tot = (1 / ssq_1) * bu_sq
        
        # Priors
        pri_r = self.g(np.squeeze(r_1), self.alph[1])[0]
        pri_U = self.h(U_1, self.lam[1])[0]
        
        return bu_tot + pri_r + pri_U

    def rep_cost_n_2_no_transform(self):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        r_0 = self.r[0]
        r_1 = self.r[1]
        U_1 = self.U[1]
        r_2 = self.r[2]
        U_2 = self.U[2]
        ssq_1 = self.ssq[1]
        ssq_2 = self.ssq[2]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        
        # 1st layer axes necessary for dot product (2D or 4D)
        bu_tdot_dims = self.bu_error_tdot_dims
        
        # Bottom up and td
        bu_v = r_0 - self.f(U1_tdot_r1)[0]
        bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_tot = (1 / ssq_1) * bu_sq
        
        td_v = r_1 - self.f(U_2.dot(r_2))[0]
        td_sq = td_v.dot(td_v)
        td_tot = (1 / ssq_2) * td_sq
        
        # Priors
        pri_r1 = self.g(np.squeeze(r_1), self.alph[1])[0]
        pri_U1 = self.h(U_1, self.lam[1])[0]
        
        '''
        this will be identical to td layer 1
        another impetus to reduce cost
        '''
        # Bottom up Layer 2
        bu_tot2 = td_tot
        
        pri_r2 = self.g(np.squeeze(r_2), self.alph[2])[0]
        pri_U2 = self.h(U_2, self.lam[2])[0]
        
        return bu_tot + td_tot + bu_tot2 + pri_r1 + pri_U1 + pri_r2 + pri_U2

    def rep_cost_n_gt_eq_3_no_transform(self):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        r_0 = self.r[0]
        r_1 = self.r[1]
        U_1 = self.U[1]
        r_2 = self.r[2]
        U_2 = self.U[2]
        ssq_1 = self.ssq[1]
        ssq_2 = self.ssq[2]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        
        # 1st layer axes necessary for dot product (2D or 4D)
        bu_tdot_dims = self.bu_error_tdot_dims
        
        # Bottom up and td
        bu_v = r_0 - self.f(U1_tdot_r1)[0]
        bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_tot = (1 / ssq_1) * bu_sq
        
        td_v = r_1 - self.f(U_2.dot(r_2))[0]
        td_sq = td_v.dot(td_v)
        td_tot = (1 / ssq_2) * td_sq
        
        # Priors
        pri_r1 = self.g(np.squeeze(r_1), self.alph[1])[0]
        pri_U1 = self.h(U_1, self.lam[1])[0]
        
        n = self.num_layers
        # For layers 2 to n-1
        bu_i = 0
        td_i = 0
        pri_ri = 0
        pri_Ui = 0
        for i in range(2,n):
            bu_v = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
            bu_sq = bu_v.dot(bu_v)
            bu_tot += (1 / self.ssq[i]) * bu_sq
            
            td_v = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
            td_sq = td_v.dot(td_v)
            td_tot += (1 / self.ssq[i+1]) * td_sq
        
            pri_r = self.g(np.squeeze(self.r[i]), self.alph[i])[0]
            pri_U = self.h(self.U[i], self.lam[i])[0]
        
            bu_i += bu_tot
            td_i += td_tot
            pri_ri += pri_r
            pri_Ui += pri_U
            
        # Final layer will only have bu term
        bu_vn = self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0]
        bu_sqn = bu_vn.dot(bu_vn)
        bu_totn = (1 / self.ssq[n]) * bu_sqn
        
        pri_rn = self.g(np.squeeze(self.r[n]), self.alph[n])[0]
        pri_Un = self.h(self.U[n], self.lam[n])[0]
        
        return bu_tot + td_tot + pri_r1 + pri_U1 + bu_i + td_i + pri_ri + pri_Ui + bu_totn + pri_rn + pri_Un
    
    def classif_cost_c1(self, label):
        # Format: -label.dot(np.log(softmax(r_n)))
        return -label.dot(np.log(self.stable_softmax(self.r[self.num_layers])))
    
    def classif_cost_c2(self, label):
        # Format: -label.dot(np.log(softmax(Uo.dot(r_n))))
        o = 'o'
        return -label.dot(np.log(self.stable_softmax(self.U[o].dot(self.r[self.num_layers])))) + self.h(self.U[o], self.lam[o])[0]
    
    def classif_cost_None(self, label):
        return 0
    
    def rn_topdown_upd_c1(self, label):
        '''
        redo for recurrent =will all be the same except rn_bar'''

        o = 'o'
        # Format: k_o / ssq_o * (label - softmax(r_n))
        c1 = (self.kr[o]/ self.ssq[o]) * (label - self.stable_softmax(self.r[self.num_layers]))
        return c1

    def rn_topdown_upd_c2(self, label):
        # Format: k_o / ssq_o * (label - softmax(Uo.dot(r_n)))
        o = 'o'
        c2 = (self.kr[o]/ self.ssq[o]) * (label - self.stable_softmax(self.U[o].dot(self.r[self.num_layers])))
        return c2
    
    def rn_topdown_upd_None(self, label):
        return 0
    
    def r_updates_n_1(self, label):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
    
    def r_updates_n_2(self, label):
        
        '''
        two layer model
        '''
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        kr_2 = self.kr[2]
        ssq_2 = self.ssq[2]
        U_2 = self.U[2]
        r_2 = self.r[2]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                            + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                            
        self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.f(self.r[1] - self.f(U_2.dot(r_2))[0])[1])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_2 / ssq_2) * self.g(r_2, self.alph[2])[1]
                                            
    def r_updates_n_gt_eq_3(self, label):
        
        n = self.num_layers
                                                
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        kr_2 = self.kr[2]
        ssq_2 = self.ssq[2]
        U_2 = self.U[2]
        r_2 = self.r[2]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
        
        # Layer 1
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                            + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
        # Layers 2 to n-1                                    
        for i in range(2,n):
            
            kr_i = self.kr[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            self.r[i] += (kr_i / ssq_i) * (U_i.T.dot(self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1])) \
                                                + (self.kr[i+1] * self.ssq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                                - (kr_i / ssq_i) * self.g(r_i, self.alph[i])[1]
    
        # Layer n
        kr_n = self.kr[n]
        ssq_n = self.ssq[n]
        U_n = self.U[n]
        r_n = self.r[n]

        self.r[n] += (kr_n / ssq_n) * (U_n.T.dot(self.f(self.r[n-1] - self.f(U_n.dot(r_n))[0])[1])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_n / ssq_n) * self.g(r_n, self.alph[n])[1]
                                                
    def U_updates_n_1(self, label):

        '''u1 will need a tensor dot
        '''
        
        '''
        check if F will work with 3d+ Us
        '''
        
        kU_1 = self.kU[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
        
        # Layer 1
        self.U[1] += (kU_1 / ssq_1) * np.einsum(self.einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                            
    def U_updates_n_gt_eq_2(self,label):
        
        kU_1 = self.kU[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
        
        # Layer 1
        self.U[1] += (kU_1 / ssq_1) * np.einsum(self.einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
        
        n = self.num_layers
        
        #i>1 - n will all be the same
        for i in range(2,n+1):
            
            kU_i = self.kU[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]

            rimin1_min_Uidotri = self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1]
            
            #i
            self.U[i] += (kU_i / ssq_i) * np.outer(rimin1_min_Uidotri, r_i) \
                        - (kU_i / ssq_i) * self.h(U_i, self.lam[i])[1]
    
    def Uo_update(self, label):
        # Format: Uo += kU_o / ssq_o * (label - softmax(Uo.dot(r_n)))
        '''
        check k/2 vs k/ssqo
        for every top down rn update, U update, V update, (place where a lr is used)
        '''
        o = 'o'
        r_n = self.r[self.num_layers]
        self.U[o] += (self.kU[o]/ self.ssq[o]) * np.outer((label - self.stable_softmax(self.U[o].dot(r_n))), r_n)
    
    def r_updates_n_1_no_transform(self, label):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]

    def r_updates_n_2_no_transform(self, label):
        
        '''
        two layer model
        '''
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        kr_2 = self.kr[2]
        ssq_2 = self.ssq[2]
        U_2 = self.U[2]
        r_2 = self.r[2]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                            + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                            
        self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.r[1] - self.f(U_2.dot(r_2))[0])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_2 / ssq_2) * self.g(r_2, self.alph[2])[1]
                                            
    def r_updates_n_gt_eq_3_no_transform(self, label):
        
        n = self.num_layers
                                                
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        kr_2 = self.kr[2]
        ssq_2 = self.ssq[2]
        U_2 = self.U[2]
        r_2 = self.r[2]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        # Layer 1
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                            + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
        # Layers 2 to n-1                                    
        for i in range(2,n):
            
            kr_i = self.kr[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            self.r[i] += (kr_i / ssq_i) * (U_i.T.dot(self.r[i-1] - self.f(U_i.dot(r_i))[0])) \
                                                + (self.kr[i+1] * self.ssq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                                - (kr_i / ssq_i) * self.g(r_i, self.alph[i])[1]

        # Layer n
        kr_n = self.kr[n]
        ssq_n = self.ssq[n]
        U_n = self.U[n]
        r_n = self.r[n]

        self.r[n] += (kr_n / ssq_n) * (U_n.T.dot(self.r[n-1] - self.f(U_n.dot(r_n))[0])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_n / ssq_n) * self.g(r_n, self.alph[n])[1]
                                                
    def U_updates_n_1_no_transform(self,label):

        '''u1 will need a tensor dot
        '''
        
        '''
        check if F will work with 3d+ Us
        '''
        
        kU_1 = self.kU[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        # Layer 1
        self.U[1] += (kU_1 / ssq_1) * np.einsum(self.einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                            
    def U_updates_n_gt_eq_2_no_transform(self,label):
        
        kU_1 = self.kU[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        # Layer 1
        self.U[1] += (kU_1 / ssq_1) * np.einsum(self.einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
        
        n = self.num_layers
        
        #i>1 - n will all be the same
        for i in range(2,n+1):
            
            kU_i = self.kU[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            rimin1_min_Uidotri = self.r[i-1] - self.f(U_i.dot(r_i))[0]
            
            #i
            self.U[i] += (kU_i / ssq_i) * np.outer(rimin1_min_Uidotri, r_i) \
                        - (kU_i / ssq_i) * self.h(U_i, self.lam[i])[1]
                        
    def Uo_update_no_transform(self, label):
        # Format: Uo += kU_o / ssq_o * (label - softmax(Uo.dot(r_n)))
        '''
        check k/2 vs k/ssqo
        for every top down rn update, U update, V update, (place where a lr is used)
        '''
        o = 'o'
        r_n = self.r[self.num_layers]
        self.U[o] += (self.kU[o]/ self.ssq[o]) * np.outer((label - self.stable_softmax(self.U[o].dot(r_n))), r_n)

    def classif_guess_c1(self, label):
        guess = np.argmax(self.stable_softmax(self.r[self.num_layers]))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
    
    def classif_guess_c2(self, label):
        guess = np.argmax(self.stable_softmax(self.U['o'].dot(self.r[self.num_layers])))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
        
    def classif_guess_None(self, label):
        return 0
    
    
class RecurrentPCC(PredictiveCodingClassifier):

    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        # This is a safeguard for now, as PCC doesn't actually have any init logic but setting attrs.
        # Initialize the base class
        super().__init__()

        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)