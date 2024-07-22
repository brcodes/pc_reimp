from parameters import constant_lr, step_decay_lr, polynomial_decay_lr

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
            save_checkpoint (None or dict): Controls checkpoint saving behavior. None to save only the final model. 
                                            Dict can have keys 'save_every' with int value N to save every N epochs, 
                                             or 'fraction' with float value 1/N to save every 1/N * Tot epochs.
            load_checkpoint (None or int): If None, do not load a checkpoint. If int, load the specified model checkpoint.
            plot (bool): If True, plot loss and accuracy.

        Notes:
            - Epoch 0 (initialized, untrained) model is always saved.
            - Final model is always saved.
            - Saving any model also saves a log.
        '''
        pass

    def evaluate(self, X, Y):
        '''
        takes input images X,
        takes labels Y,
        evaluates the model classification accuracy, pe
        
        opt: plot pe
        '''
        pass
    
    def predict(self, X):
        '''
        takes input images X,
        predicts the image (plot topmost representation and pe)
        '''
        pass


class TiledStaticPCC(PredictiveCodingClassifier):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for static PCC

class TiledRecurrentPCC(PredictiveCodingClassifier):
    def __init__(self):
        super().__init__()

    # Override methods as necessary for recurrent PCC
