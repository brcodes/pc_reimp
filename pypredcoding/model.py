from parameters import constant_lr, step_decay_lr, polynomial_decay_lr

def save_results(results, output_file):
    with open(output_file, 'wb') as f:
        f.write(str(results))

class PredictiveCodingClassifier:
    def __init__(self):
        pass

    def train(self, X, Y, save_checkpoint=None, load_checkpoint=None, plot=False):
        '''
        Trains the model using input images and labels, with options for checkpointing and plotting.

        Parameters:
            X (np.array): Input images, shape=(num_imgs, numxpxls * numypxls) [non-tiled] or 
                          (num_imgs * numtlsperimg, numtlxpxls * numtlypxls) [tiled].
            Y (np.array): Labels, shape=(num_imgs, num_classes).
            save_checkpoint (None or dict, optional): Controls checkpoint saving behavior. None to save only the final model. 
                                            Dict can have keys 'save_every' with int value N to save every N epochs, 
                                             or 'fraction' with float value 1/N to save every 1/N * Tot epochs.
            load_checkpoint (None or int, optional): If None, do not load a checkpoint. If Int != -1 load the specified model checkpoint.
                                                    If Int == -1, load the latest model checkpoint.
            plot (bool, optional): If True, plot loss and accuracy.

        Notes:
            - Epoch 0 (initialized, untrained) model is always saved.
            - Final model is always saved.
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

    # Override methods as necessary for static PCC
    
class TiledStaticPCC(StaticPCC):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for static PCC
    
class RecurrentPCC(PredictiveCodingClassifier):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC

class TiledRecurrentPCC(RecurrentPCC):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC
