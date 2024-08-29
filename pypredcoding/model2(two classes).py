from parameters import constant_lr, step_decay_lr, polynomial_decay_lr
import numpy as np
from functools import partial

from datetime import datetime
import pickle
from os import path


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
        name: str: name of experiment
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
        self.config_from_attributes()
        
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
        num_layers = self.num_layers
        n = num_layers
        for i in range(1, n + 1):
            if i == n:
                self.r[i] = self.prior_dist(size=(self.output_lyr_size))
            else:
                self.r[i] = self.prior_dist(size=(self.hidden_lyr_sizes[i - 1]))
            print(f'r{i} shape: {self.r[i].shape}')
        
        # Initiate Us
        # U1 is going to be a little bit different
        U1_size = tuple(list(self.input_size) + list(self.r[1].shape))
        self.U[1] = self.prior_dist(size=U1_size)
        # U2 through Un
        for i in range(2, n + 1):
            Ui_size = (self.r[i-1].shape[0], self.r[i].shape[0])
            self.U[i] = self.prior_dist(size=Ui_size)
            print(f'U{i} shape: {self.U[i].shape}')
        if self.classif_method == 'c2':
            Uo_size = (self.num_classes, self.output_lyr_size)
            self.U['o'] = self.prior_dist(size=Uo_size)
            print(f'Uo shape: {self.U['o'].shape}')
            
        # Initiate Jr, Jc, and accuracy (diagnostics) for storage, print, plot
        epoch_n = self.epoch_n
        self.Jr = {i: [0] * (epoch_n + 1) for i in range(n)}
        self.Jc = {i: [0] * (epoch_n + 1) for i in range(n)}
        self.accuracy = [0] * (epoch_n + 1)