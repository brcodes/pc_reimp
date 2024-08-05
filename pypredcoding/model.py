from parameters import constant_lr, step_decay_lr, polynomial_decay_lr
import numpy as np
from functools import partial
        
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
        self.update_method_dict = {'rU_niters': self.update_method_rUniters,
                                    'r_niters_U': self.update_method_r_niters_U,
                                    'r_eq_U': self.update_method_r_eq_U}
        self.rn_topdown_cost_dict = {'c1': self.rn_topdown_cost_c1,
                                    'c2': self.rn_topdown_cost_c2,
                                    'None': self.rn_topdown_cost_None}
        
        self.component_updates = [self.r_updates, self.U_updates]
        
        # Transforms and priors
        self.f = self.act_fxn_dict[self.activ_func]
        self.g = self.prior_cost_dict[self.priors]
        self.h = self.prior_cost_dict[self.priors]
        
        '''
        dict fill later
        '''
        '''
        change rpcc learning rates to kr ku kv, etc.
        npt alpha beta gamma
        '''
        # Prior parameters
        self.alph = {}
        self.lam = {}
        
        # None inits are to be received from assignment
        # Rough types are given
        # Size
        self.num_layers = None
        self.input_size = (None)
        self.hidden_layer_sizes = [None]
        self.output_layer_size = None
        
        # Training
        self.dataset_train = 'None'
        self.update_method = {None:None}
        self.batch_size = None
        self.epoch_n = None
        self.save_checkpoint = None
        self.load_checkpoint = None
        self.plot_train = None
        
        # Evaluation
        self.dataset_eval = 'None'
        self.plot_eval = 'None'
        
        # Prediction
        self.dataset_pred = 'None'
        self.plot_pred = 'None'
        
        # Model components, V's specific to recurrent subclass
        self.r = {}
        self.U = {}
        # Learning rates
        self.kr = {}
        self.kU = {}
        
        # Layer variances (just a divisor for learning rates, functionally)
        # All ssqs should be 1, could experiment with other (dynamic?) values later
        self.ssq = {}
        
        
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
        
        # Uo is an output weight matrix that learns to relate r_n to the label        
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.output_lyr_size)
            self.U['o'] = self.prior_dist(size=Uo_size)
    
        
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
        F = np.eye(len(f))
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
    
    def rn_topdown_cost_c1(self, label):
        '''
        see if ssqo or 2 in denom
        '''
        o = 'o'
        # Format: k_o / ssq_o * (label - softmax(r_n))
        c1 = (self.kr[o]/ self.ssq[o]) * (label - self.softmax(self.self.r[self.num_layers]))
        return c1
    
    def rn_topdown_cost_c2(self, label):
        # Format: k_o / ssq_o * (label - softmax(Uo.dot(r_n)))
        o = 'o'
        c2 = (self.kr[o]/ self.ssq[o]) * (label - self.softmax(self.U[o].dot(self.r[self.num_layers])))
        return c2
    
    def rn_topdown_cost_None(self):
        return 0
        
    def r_updates(self, label):
        '''
        test
        
        verify kn / 2 paradigm
        '''
        n = self.num_layers
        kr_n = self.kr[n]
        ssq_n = self.ssq[n]
        U_n = self.U[n]
        r_n = self.r[n]
        
        for i in range(1,n):
            
            kr_i = self.kr[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            #i
            # ri += ki/ssqi UiT F(ri-1 - f(Ui ri)) (bottom up component)
            # + ki+1/ssqi+1 (f(Ui+1 ri+1) - ri) (top down component)
            # - ki/2 g'(ri) (prior component)
            self.r[i] += (kr_i / ssq_i) * (U_i.T.dot(self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1])) + \
                                                + (self.kr[i+1] * self.ssq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                                - (kr_i / 2) * self.g(r_i, self.alph[i])[1]
        
        #n
        # rn += kn/ssqn UnT F(rn-1 - f(Un rn)) (bottom up component)
        # + C1/C2/None (top down component)
        # - kn/2 g'(rn) (prior component)
        self.r[n] += (kr_n / ssq_n) * (U_n.T.dot(self.f(self.r[n-1] - self.f(U_n.dot(r_n))[0])[1])) + \
                                                + self.rn_topdown_cost_dict[self.classif_method](label) \
                                                - (kr_n / 2) * self.g(r_n, self.alph[n])[1]
    
    def U_updates(self):
        pass
    
    '''
    make a V_updates for the recurrent subclass
    '''
    
    def update_method_rUniters(self, niters, label):
        '''
        Li def: 30
        '''
        
        for i in range(niters):
            self.r_updates(label)
            self.U_updates(label)
        
    def update_method_r_niters_U(self, niters, label):
        '''
        Rogers/Brown def: 100
        '''
        
        for i in range(niters):
            self.r_updates(label)
        self.U_updates(label)

    def update_method_r_eq_U(self, stop_criterion, label):
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
        
        initial_norms = [np.linalg.norm(self.r[i]) for i in range(1, self.num_layers + 1)]
        diffs = [float('inf')] * self.num_layers  # Initialize diffs to a large number
        
        while any(diff > stop_criterion for diff in diffs):
            prev_r = {i: self.r[i].copy() for i in range(1, self.num_layers + 1)}  # Copy all vectors to avoid reference issues
            self.r_updates(label)
            
            for i in range(1, self.num_layers + 1):
                post_r = self.r[i]
                diff_norm = np.linalg.norm(post_r - prev_r[i])
                diffs[i-1] = (diff_norm / initial_norms[i-1]) * 100  # Calculate the percentage change
        
        self.U_updates(label)
    
    def save_results(self, results, output_file):
        with open(output_file, 'wb') as f:
            f.write(str(results))
            
    def train(self, X, Y, save_checkpoint=None, plot=False):
        '''
        Trains the model using input images and labels, with options for checkpointing and plotting.

        Parameters:
            X (np.array): Input images, shape=(num_imgs, numxpxls * numypxls) [non-tiled] or 
                          (num_imgs * numtlsperimg, numtlxpxls * numtlypxls) [tiled].
            Y (np.array): Labels, shape=(num_imgs, num_classes).
            save_checkpoint (None or dict, optional): Controls checkpoint saving behavior. None to save only the final model. 
                                            Dict can have keys 'save_every' with int value N to save every N epochs, 
                                             or 'fraction' with float value 1/N to save every 1/N * Tot epochs.
            plot (bool, optional): If True, plot loss and accuracy.

        Notes:
            - Epoch 0 (initialized, untrained) model is always saved. (models/checkpoints/)
            - Final model is always saved. (models/)
            - Saving any model also saves a log.
        '''
        # Parse update method
        # e.g. self.update_method = {'rU_niters' (update method name): 30 (update method number)}
        # See config for more.
        update_method_name = next(iter(self.update_method))
        update_method_number = self.update_method[update_method_name]
        
        for e in epochs:
            for i in images:
                self.r[0] = input
                label = Y[i]
                
                # Update components
                self.update_method_dict[update_method_name](update_method_number, label)
                
        pass

    def evaluate(self, X, Y, plot=None):
        '''
        Evaluates the model's classification accuracy using input images and labels.

        Parameters:
            X (np.array): Input images to be evaluated by the model.
            Y (np.array): True labels for the input images.
            plot (None or Str, optional): Assigns prediction error (pe) plotting behavior. None to plot nothing. 
                                Str can be 'first' to plot model response to first image, f'rand{N}' to plot responses to N random 
                                images, or 'all' for those of all images.
        '''
        pass
    
    def predict(self, X, plot=None):
        '''
        Predicts the class of input images and optionally plots the topmost representation

        Parameters:
            X (np.array): Input images for prediction.
            plot (None or Str, optional): Assigns topmost representation and prediction error (pe) plotting behavior. None to plot nothing. 
                                Str can be 'first' to plot model response to first image, f'rand{N}' to plot responses to N random 
                                images, or 'all' for those of all images.
        '''
        
        pass


class StaticPCC(PredictiveCodingClassifier):
    
    def __init__(self):
        super().__init__()
        
        """
        SPCC can take:
        1/2D input (flat_image_area,) or (1,flat_image_area): flattened
        2D input (imageypxls, imagexpxls): unflattened
        
        unflattened preserves spatial information, especially important in image processing
        and more biologically realistic, as we see entire '2D' fields on surfaces
        if flat works, though, great. it'll be faster, and speaks to the power of the PC model.
        """
        
    

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
        
    def validate_attributes(self):

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
