from parameters import constant_lr, step_decay_lr, polynomial_decay_lr
import numpy as np
from functools import partial
from no_transform_updates_and_costs import r_updates_n_1_no_transform, r_updates_n_2_no_transform, r_updates_n_gt_eq_3_no_transform, \
                                            U_updates_n_1_no_transform, U_updates_n_gt_eq_2_no_transform, \
                                            rep_cost_n_1_no_transform, rep_cost_n_2_no_transform, rep_cost_n_gt_eq_3_no_transform

        
class PredictiveCodingClassifier:

    def __init__(self):
        
        '''
        I think not unique to any subclass, so I'll put it here.
        '''
        # Choices for transformation functions, priors
        self.act_fxn_dict = {'linear': self.linear_transform,
                                'tanh': self.tanh_transform}
        self.prior_cost_dict = {'gaussian': self.gaussian_prior_costs, 
                                'kurtotic': self.kurtotic_prior_costs}
        self.prior_dist_dict = {'gaussian': partial(np.random.normal, loc=0, scale=1),
                                'kurtotic': partial(np.random.laplace, loc=0.0, scale=0.5)}
    
        '''
        change rpcc learning rates to kr ku kv, etc.
        npt alpha beta gamma
        '''
        # None/wrapped None inits are to be received from assignment
        # Rough types are given
        # Model
        self.model_type = 'None'
        self.tiled = None
        # Size
        self.num_layers = None
        self.input_size = (None)
        self.hidden_layer_sizes = [None]
        self.output_layer_size = None
        # Model components, V's specific to recurrent subclass
        self.r = {}
        self.U = {}
        # Training
        self.dataset_train = 'None'
        self.classif_method = 'None'
        self.activ_func = 'None'
        self.priors = 'None'
        self.update_method = {None:None}
        self.batch_size = None
        self.epoch_n = None
        self.save_checkpoint = None
        self.load_checkpoint = None
        self.online_diagnostics = None
        self.plot_train = None
        # Learning rates
        self.kr = {None:None}
        self.kU = {None:None}
        # Layer variances (just a divisor for learning rates, functionally)
        # All ssqs should be 2 (fulfills gradient coefficient in some cases), could experiment with other (dynamic?) values later
        self.ssq = {None:None}
        # Prior parameters
        self.alph = {None:None}
        self.lam = {None:None}
        # Evaluation
        self.dataset_eval = 'None'
        self.plot_eval = 'None'
        # Prediction
        self.dataset_pred = 'None'
        self.plot_pred = 'None'
        # Diagnostics
        self.Je = {}
        self.Jc = {}
        self.accuracy = []
        
    def set_model_attributes(self, params):
        '''
        Set model attributes from a dictionary.
        This will set a bunch of external attributes too,
        which will serve no other purpose than to recount the last experiment run on the model. e.g. name, train, notes.
        '''
        for key, value in params.items():
            setattr(self, key, value)
        self.config_from_attributes()
        
    def config_from_attributes(self):
        '''
        Configure the PCC based on the set attributes.
        '''
        # Transforms and priors
        self.f = self.act_fxn_dict[self.activ_func]
        self.g = self.prior_cost_dict[self.priors]
        self.h = self.prior_cost_dict[self.priors]

        # Initiate rs
        # Layer 1 until n will be the same
        # r0 is the input, assigned in subclass
        for lyr in range(1,self.num_layers+1):
            self.r[lyr] = self.prior_dist(size=(self.hidden_lyr_sizes[lyr-1]))
            if lyr == self.num_layers:
                self.r[lyr] = self.prior_dist(size=(self.output_lyr_size))
                
        # Initiate Us
        # Layer 2 until n will be the same
        # U1 relates r0 to r1, assigned in subclass
        for lyr in range(2,self.num_layers+1):
            # Ui>1 is going to be 2D, and the size is the same as (the r of the previous layer, r of current layer)
            Ui_gt_1_size = (self.r[lyr-1].shape, self.r[lyr].shape)
            self.U[lyr] = self.prior_dist(size=Ui_gt_1_size)
        
        # Uo is an output weight matrix that learns to relate r_n to the label, size of (num_classes, topmost_r)
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.output_lyr_size)
            self.U['o'] = self.prior_dist(size=Uo_size)
            
        # Initiate Je, Jc and accuracy (diagnostics) storage for print/log, plot
        self.Je = {i: [0] * (self.epoch_n + 1) for i in range(self.num_layers)}
        self.Jc = {i: [0] * (self.epoch_n + 1) for i in range(self.num_layers)}
        self.accuracy = [0] * (self.epoch_n + 1)
        
    # r, U or V prior functions
    def gaussian_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior.
        """
        g_or_h = alph_or_lam * np.square(r_or_U).sum()
        gprime_or_hprime = 2 * alph_or_lam * r_or_U
        return (g_or_h, gprime_or_hprime)

    def kurtotic_prior_costs(self, r_or_U=None, alph_or_lam=None):
        """
        Takes an argument pair of either r & alpha, or U & lambda, and returns
        a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
        """
        g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
        gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
        return (g_or_h, gprime_or_hprime)

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

    def prior_dist(self):
        return self.prior_dist_dict[self.priors]
    
    def softmax(self, vector, k=1):
        """
        Compute the softmax function of a vector.
        Parameters:
        - vector: numpy array or list
            The input vector.
        - k: float, optional (default=1)
            The scaling factor for the softmax function.
        Returns:
        - softmax_vector: numpy array
            The softmax of the input vector.
        """
        exp_vector = np.exp(k * vector)
        softmax_vector = exp_vector / np.sum(exp_vector)
        return softmax_vector
    
    def rn_topdown_term_None(self):
            return 0
        
    def classif_cost_None(self):
        return 0
    
    '''
    make a V_updates for the recurrent subclass
    it will take label but do nothing with it
    
    '''
    
    def update_method_rWniters(self, niters, label, component_updates):
        '''
        Li def: 30
        '''
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        for i in range(niters):
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
        '''
        Rogers/Brown def: 0.05
        '''
        
        '''
        i diffs, or just one?
        '''
        '''
        r is a dictionary e.g.
        
        r[1] 32
        r[2] 128
        r[3] 212'''
        
        r_updates = component_updates[0]
        # Can be U/Uo or U/Uo,V
        weight_updates = component_updates[1:]
        num_weight_updates = len(weight_updates)
        range_num_weight_updates = range(num_weight_updates)
        
        num_layers = self.num_layers
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, num_layers + 1)]
        diffs = [float('inf')] * num_layers  # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, num_layers + 1)}  # Copy all vectors to avoid reference issues
            r_updates(label)
            
            for i in range(1, num_layers + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change
        
        for w in range_num_weight_updates:
            # For as many weight sets are there are to update, update them.
            weight_updates[w](label) 
    
    def save_results(self, results, output_file):
        with open(output_file, 'wb') as f:
            f.write(str(results))
            
    def train(self, X, Y, save_checkpoint=None, online_diagnostics=False, plot=False):
        '''
        Trains the model using input images and labels, with options for checkpointing and plotting.

        Parameters:
            X (np.array): Input images, shape=(num_imgs, numxpxls * numypxls) [non-tiled] or 
                          (num_imgs * numtlsperimg, numtlxpxls * numtlypxls) [tiled].
            Y (np.array): Labels, shape=(num_imgs, num_classes).
            save_checkpoint (None or dict, optional): Controls checkpoint saving behavior. None to save only the final model. 
                                            Dict can have keys 'save_every' with int value N to save every N epochs, 
                                             or 'fraction' with float value 1/N to save every 1/N * Tot epochs.
            online_diagnostics (bool, optional): If True, store and print diagnostics at each epoch. Must be on for plot.
            plot (bool, optional): If True, plot loss and accuracy after training. online_diagnostics must be on.

        Notes:
            - Epoch 0 (initialized, untrained) model is always saved. (models/checkpoints/)
            - Final model is always saved. (models/)
            - Saving any model also saves a log.
        '''
        # Params
        epoch_n = self.epoch_n
        num_imgs = X.shape[0]
        
        # Parse update method
        # e.g. self.update_method = {'rW_niters' (update method name): 30 (update method number)}
        # See config for more.
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        update_components = partial(self.update_method_dict[update_method_name], update_method_number)
        
        # Representation cost
        rep_cost = self.rep_cost
        # Classification
        classif_method = self.classif_method
        classif_cost = self.classif_cost_dict[classif_method]
        # Accuracy
        evaluate = partial(self.evaluate, plot=None)
        
        
        # Rep cost, other diagnostics at "Epoch 0" (initialized, untrained)
        if online_diagnostics:
            # Epoch 0
            epoch = 0
            for i in range(num_imgs):
                input = X[i]
                self.r[0] = input
                label = Y[i]
                # Calculate loss and accuracy
                Je = rep_cost()
                Jc = classif_cost(label)
            accuracy = evaluate(X, Y)
            self.accuracy[epoch] = accuracy
        
        for e in range(epoch_n):
            epoch = e + 1
            print(f'Epoch {epoch}')
            for i in range(num_imgs):
                input = X[i]
                self.r[0] = input
                label = Y[i]
                
                # Update components
                # e.g. r, U/Uo, V
                update_components(label=label)
                
                '''
                don't worry about checkpointing now
                '''
                
        if plot:
            # Plot loss and accuracy
            
            
            
            pass

    def evaluate(self, X, Y, plot=None):
        '''
        Evaluates the model's classification accuracy using input images and labels.

        Parameters:
            X (np.array): Input images to be evaluated by the model.
            Y (np.array): True labels for the input images.
            plot (None or Str, optional): Assigns prediction error (pe) plotting behavior. None to save no PE data and plot nothing. 
                                Str can be 'first' to plot model response to first image, f'rand{N}' to plot responses to N random 
                                images, or 'all' for those of all images.
        '''
        return accuracy
    
    def predict(self, X, plot=None):
        '''
        Predicts the class of input images and optionally plots the topmost representation

        Parameters:
            X (np.array): Input images for prediction.
            plot (None or Str, optional): Assigns topmost representation and prediction error (pe) plotting behavior. None to plot nothing. 
                                Str can be 'first' to plot model response to first image, f'rand{N}' to plot responses to N random 
                                images, or 'all' for those of all images.
        '''
        
        return class_predictions


class StaticPCC(PredictiveCodingClassifier):
    
    def __init__(self, base_instance: PredictiveCodingClassifier):
        
        """
        SPCC can take:
        1/2D input (flat_image_area,) or (1,flat_image_area): flattened
        2D input (imageypxls, imagexpxls): unflattened
        
        unflattened preserves spatial information, especially important in image processing
        and more biologically realistic, as we see entire '2D' fields on surfaces
        if flat works, though, great. it'll be faster, and speaks to the power of the PC model.
    
        """
        # Copy attributes from the base instance
        self.__dict__.update(base_instance.__dict__)
        
        '''
        test
        '''
        # No F in the update equations. only f().
        self.r_updates_n_1 = r_updates_n_1_no_transform
        self.r_updates_n_2 = r_updates_n_2_no_transform
        self.r_updates_n_gt_eq_3 = r_updates_n_gt_eq_3_no_transform
        self.U_updates_n_1 = U_updates_n_1_no_transform
        self.U_updates_n_gt_eq_2 = U_updates_n_gt_eq_2_no_transform
        self.rep_cost_n_1 = rep_cost_n_1_no_transform
        self.rep_cost_n_2 = rep_cost_n_2_no_transform
        self.rep_cost_n_gt_eq_3 = rep_cost_n_gt_eq_3_no_transform
        
        '''
        test
        '''
        
        if self.num_layers == 1:
            self.r_updates = self.r_updates_n_1
            self.U_updates = self.U_updates_n_1
            self.rep_cost = self.rep_cost_n_1
        elif self.num_layers == 2:
            self.r_updates = self.r_updates_n_2
            self.U_updates = self.U_updates_n_gt_eq_2
            self.rep_cost = self.rep_cost_n_2
        elif self.num_layers >= 3:
            self.r_updates = self.r_updates_n_gt_eq_3
            self.U_updates = self.U_updates_n_gt_eq_2
            self.rep_cost = self.rep_cost_n_gt_eq_3
        else:
            raise ValueError("Number of layers must be at least 1.")
            
        self.component_updates = [self.r_updates, self.U_updates]
        if self.classif_method == 'c2':
            self.component_updates.append(self.Uo_update)
            
        self.update_method_dict = {'rW_niters': partial(self.update_method_rWniters, component_updates=self.component_updates),
                                    'r_niters_W': partial(self.update_method_r_niters_W, component_updates=self.component_updates),
                                    'r_eq_W': partial(self.update_method_r_eq_W, component_updates=self.component_updates)}
        
        # Subclass needs topdown and cost None terms to exist before assignment
        self.rn_topdown_term_dict = {'c1': self.rn_topdown_term_c1,
                                    'c2': self.rn_topdown_term_c2,
                                    None: self.rn_topdown_term_None}
        
        self.classif_cost_dict = {'c1': self.classif_cost_c1,
                                'c2': self.classif_cost_c2,
                                None: self.classif_cost_None}
            
        def rep_cost_n_1(self):
            '''
            Uses current r/U states to compute the least squares portion of the error
            (concerned with accurate reconstruction of the input).
            '''
            
            '''
            change layer 1 to match tensordot form
            '''
            E = 0
            # LSQ cost
            PE_list = []
            
            for i in range(0,len(self.r)-1):
                v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                vTdotv = v.T.dot(v)
                E = E + ((1 / self.ssq[i+1]) * vTdotv)[0,0]

                # Also calulate prediction error for each layer
                PE = np.sqrt(vTdotv)
                PE_list.append(PE)

            # priors on r[1],...,r[n]; U[1],...,U[n]
            for i in range(1,len(self.r)):
                E = E + (self.h(self.U[i],self.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.alph[i])[0])

            return (E, PE_list)
        
        def rep_cost_n_2(self):

            return (E, PE_list)
        
        def rep_cost_n_gt_eq_3(self):
            return (E, PE_list)
        
        def classif_cost_c1(self, label):
            # Format: -label.dot(np.log(softmax(r_n)))
            return -label.dot(np.log(self.softmax(self.r[self.num_layers])))
        
        def classif_cost_c2(self, label):
            # Format: -label.dot(np.log(softmax(Uo.dot(r_n))))
            o = 'o'
            return -label.dot(np.log(self.softmax(self.U[o].dot(self.r[self.num_layers])))) + self.h(self.U[o], self.lam[o])[0]
        
        def rn_topdown_term_c1(self, label):
            '''
            redo for recurrent =will all be the same except rn_bar'''
            '''
            see if ssqo or 2 in denom
            '''
            o = 'o'
            # Format: k_o / ssq_o * (label - softmax(r_n))
            c1 = (self.kr[o]/ self.ssq[o]) * (label - self.softmax(self.r[self.num_layers]))
            return c1
    
        def rn_topdown_term_c2(self, label):
            # Format: k_o / ssq_o * (label - softmax(Uo.dot(r_n)))
            o = 'o'
            c2 = (self.kr[o]/ self.ssq[o]) * (label - self.softmax(self.U[o].dot(self.r[self.num_layers])))
            return c2
        
        def set_U1_operation_dims(self, U1_size, input_size):
            # Transpose dims
            ndims_U1 = len(U1_size)
            range_ndims_U1 = range(ndims_U1)
            last_dim_id_U1 = range_ndims_U1[-1]
            nonlast_dim_ids_U1 = range_ndims_U1[:-1]
            transpose_dims_U1 = tuple([last_dim_id_U1] + list(nonlast_dim_ids_U1))
            self.U1_transpose_dims = transpose_dims_U1
            
            # Tensordot dims
            # U1T, last n-1
            nonfirst_dim_ids_U1 = range_ndims_U1[1:]
            self.U1T_tdot_dims = list(nonfirst_dim_ids_U1)
            
            # Input dims, all
            ndims_input = len(input_size)
            range_ndims_input = range(ndims_input)
            self.input_min_U1tdotr1_tdot_dims = list(range_ndims_input)

        def r_updates_n_1(self, label):
            '''
            move to static eventually, as well as update_Component assignment
            '''
                
            kr_1 = self.kr[1]
            ssq_1 = self.ssq[1]
            U_1 = self.U[1]
            r_1 = self.r[1]
            
            #U1 operations
            U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
            U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
            input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
            
            self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                                    + self.rn_topdown_term_dict[self.classif_method](label) \
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
            U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
            U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
            input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
            
            self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                                + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                                - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                                
            self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.f(self.r[1] - self.f(U_2.dot(r_2))[0])[1])) \
                                                    + self.rn_topdown_term_dict[self.classif_method](label) \
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
            U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
            U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
            input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
            
            # Layer 1
            self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
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
                                                    + self.rn_topdown_term_dict[self.classif_method](label) \
                                                    - (kr_n / ssq_n) * self.g(r_n, self.alph[n])[1]
                                                    
        def U_updates_n_1(self):
    
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
            self.U[1] += (kU_1 / ssq_1) * np.outer(input_min_U1tdotr1, r_1) \
                            - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                                
        def U_updates_n_gt_eq_2(self):
            
            kU_1 = self.kU[1]
            ssq_1 = self.ssq[1]
            U_1 = self.U[1]
            r_1 = self.r[1]
            
            #U1 operations
            U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
            input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
            
            # Layer 1
            self.U[1] += (kU_1 / ssq_1) * np.outer(input_min_U1tdotr1, r_1) \
                            - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
            
            n = self.num_layers
            
            #i>1 - n will all be the same
            for i in range(1,n+1):
                
                kU_i = self.kU[i]
                ssq_i = self.ssq[i]
                r_i = self.r[i]
                U_i = self.U[i]
                
                #i
                self.U[i] += (kU_i / ssq_i) * np.outer((self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1]), r_i) \
                            - (kU_i / ssq_i) * self.h(U_i, self.lam[i])[1]
        
        def Uo_update(self, label):
            # Format: Uo += kU_o / ssq_o * (label - softmax(Uo.dot(r_n)))
            '''
            check k/2 vs k/ssqo
            for every top down rn update, U update, V update, (place where a lr is used)
            '''
            o = 'o'
            r_n = self.r[self.num_layers]
            self.U[o] += (self.kU[o]/ self.ssq[o]) * np.outer((label - self.softmax(self.U[o].dot(r_n))), r_n)
        

    # Override methods as necessary for static PCC
    
class TiledStaticPCC(StaticPCC):
    
    def __init__(self):
        super().__init__()
        
        """
        tSPCC can take:
        2D input (num_tiles, flat_tile_area): flattened
        4D input (num_tile_rows, num_tile_columns, numtlypxls, numtlxpxls): unflattened
        
        unflattened preserves spatial information, especially important in image processing
        and more biologically realistic, as we see entire '2D' fields on surfaces
        if flat works, though, great. it'll be faster, and speaks to the power of the PC model.
        """
        # For non-flat 212 pseudospectrograms, 1 input is 4,4,24,36
        # That's a grid of 4x4 tiles, each tile is 24x36 pixels
        
        # None inits are to be received from assignment
        if len(self.input_size) == 2:
            self.flat_input = True
        elif len(self.input_size) == 4:
            self.flat_input = False
        else:
            raise ValueError("Input size must be 2D or 4D.")
        
        # Tiles
        if self.flat_input:
            self.num_tiles = self.input_size[0]
        else:
            self.num_tiles = self.input_size[0] * self.input_size[1]
        
        # Initiate r0
        # Layer 0 will be the input, filled during training, testing, etc.
        self.r[0] = np.zeros(self.input_size)
                
        # Initiate U1
        # Layer 1 will expand based on the shape of r0
        # Could be 3D or 5D
        U1_size = tuple(list(self.input_size) + list(self.r[1].shape))
        self.U[1] = self.prior_dist(size=U1_size)
        # Get transpose and tensordot axes ready for U1 operations
        self.set_U1_operation_dims(U1_size, self.input_size)

        
    def validate_attributes(self):
        # config file should have all necessary attributes
        # think about improper combinations

        pass
        # Add more validation checks as needed

    # Override methods as necessary for static PCC
    
class RecurrentPCC(PredictiveCodingClassifier):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC

class TiledRecurrentPCC(RecurrentPCC):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC
