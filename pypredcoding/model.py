from parameters import constant_lr, step_decay_lr, polynomial_decay_lr
import numpy as np

# r, U or V prior functions
def gauss_prior(r_or_U=None, alph_or_lam=None):
    """
    Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior.
    """
    g_or_h = alph_or_lam * np.square(r_or_U).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)

def kurt_prior(r_or_U= None, alph_or_lam=None):
    """
    Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
    """
    g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    return (g_or_h, gprime_or_hprime) 

# Activation functions
def linear_trans(U_dot_r):
    """
    Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Linear transformation.
    """
    f = U_dot_r
    F = np.eye(len(f))
    return (f, F)
    
def tanh_trans(U_dot_r):
    """
    Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Tanh transformation.
    """
    f = np.tanh(U_dot_r)
    F = np.diag(1 - f.flatten()**2)
    return (f, F)

# Helpers
def softmax(vector, k=1):
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

def save_results(results, output_file):
    with open(output_file, 'wb') as f:
        f.write(str(results))
        
class PredictiveCodingClassifier:
    def __init__(self):
        
        '''
        I think not unique to any subclass, so I'll put it here.
        '''
        # Choices for transformation functions, priors
        act_fxn_dict = {'linear':linear_trans,'tanh':tanh_trans}
        prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # Transforms and priors
        self.f = act_fxn_dict[self.activ_func]
        self.g = prior_dict[self.priors]
        self.h = prior_dict[self.priors]
        
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
        3D input (num_tiles, numtlypxls, numtlxpxls): unflattened
        
        unflattened preserves spatial information, especially important in image processing
        and more biologically realistic, as we see entire '2D' fields on surfaces
        if flat works, though, great. it'll be faster, and speaks to the power of the PC model.
        """
        
        self.flat_input = False
        
        self.input_size = (1, 1, 1)
        
        self.num_tiles = self.input_size[0]
        
        self.hidden_lyr_sizes = []
        
        self.all_r = {}
        self.all_U = {}
        
        # Initiate rs
        # Layer 0 will be the input
        self.r[0] = np.zeros(self.input_size)
        
        # Layer 1 until n will be the same
        for lyr in range(1,self.num_layers+1):
            
            if self.priors == 'kurtotic':
                self.r[lyr] = np.random.laplace(loc=0.0, scale=0.5, size=(self.hidden_lyr_sizes[lyr-1]))
                if lyr == self.num_layers:
                    self.r[lyr] = np.random.laplace(loc=0.0, scale=0.5, size=(self.output_lyr_size))
                    
            elif self.priors == 'gaussian':
                self.r[lyr] = np.random.standard_normal(size=(self.hidden_lyr_sizes[lyr-1]))
                if lyr == self.num_layers:
                    self.r[lyr] = np.random.standard_normal(size=(self.output_lyr_size))
                    
        # Initiate Us 
        for lyr in range(1,self.num_layers+1):        
            self.U[lyr] = np.random.laplace(loc=0.0, scale=0.5, size=((self.hidden_lyr_sizes[lyr-1])))

        
    def validate_attributes(self):
        if self.flat_input is False and len(self.input_size) != 3:
            raise ValueError("Unflattened input must be 3D.")
        elif self.flat_input is True and len(self.input_size) != 2:
            raise ValueError("Flattened input must be 2D.")
        # Add more validation checks as needed
        
    import numpy as np

    # Assuming U1 is your 3D matrix of shape (84, 132, 32)
    # and r1 is your 1D vector of shape (32,)

    # You can use np.tensordot to perform the dot product along the last axis
    result = np.tensordot(U1, r1, axes=([-1], [0]))

    # Now, result is a 2D matrix of shape (84, 132)


    # Override methods as necessary for static PCC
    
class RecurrentPCC(PredictiveCodingClassifier):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC

class TiledRecurrentPCC(RecurrentPCC):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC
