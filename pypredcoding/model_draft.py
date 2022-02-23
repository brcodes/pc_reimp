import numpy as np
from matplotlib import pyplot as plt
from learning import *
from functools import partial
import math
import data
import cv2
from parameters import ModelParameters
from sys import exit

"""
A Predictive Coding Classifier according to the mathematical dictates of Rao & Ballard 1999.
"""

#### Functions that need to be outside of the PCC class
#### e.g. to be called in a dictionary before training, avoiding in-loop if statments

### Activation functions

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

### r, U prior functions

def gauss_prior(r_or_U, alph_or_lam):
    """
    Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior.
    """

    g_or_h = alph_or_lam * np.square(r_or_U).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)

def kurt_prior(r_or_U, alph_or_lam):
    """
    Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior.
    """

    g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    return (g_or_h, gprime_or_hprime)


class PredictiveCodingClassifier:
    def __init__(self, parameters):

        """
        Model layer architecture not initialized until model.train() called
        """

        self.p = parameters

        # All the representations (including the image r[0] which is not trained)
        self.r = {}
        # Synaptic weights controlling reconstruction in the network
        self.U = {}

        # Num hidden layers (Li case: 2)
        self.num_hidden_lyrs = len(self.p.hidden_sizes)
        # Number of non-input layers for model architecture init: always num hidden layers + 1 output layer (Li case: 3)
        self.num_nonin_lyrs = self.num_hidden_lyrs + 1
        # Total num includes input "layer" (Li case: 4)
        self.num_tot_lyrs = len(self.num_nonin_lyrs) + 1

        # Choices for transformation functions, priors
        self.act_fxn_dict = {'lin':linear_trans,'tan':tanh_trans}
        self.prior_dict = {'gaus':gauss_prior, 'kurt':kurt_prior}

        # Transforms and priors
        self.f = self.act_fxn_dict[self.p.act_fxn]
        self.g = self.prior_dict[self.p.r_prior]
        self.h = self.prior_dict[self.p.U_prior]

        # learning rate functions (can't figure out how to dispatch this)
        lr_r = list(self.p.k_r_sched.keys())[0]
        if lr_r == 'constant':
            self.k_r_lr = partial(constant_lr,initial=self.p.k_r_sched['constant']['initial'])
        elif lr_r == 'step':
            self.k_r_lr = partial(step_decay_lr,initial=self.p.k_r_sched['step']['initial'],drop_every=self.p.k_r_sched['step']['drop_every'],drop_factor=self.p.k_r_sched['step']['drop_factor'])
        elif lr_r == 'poly':
            self.k_r_lr = partial(polynomial_decay_lr,initial=self.p.k_r_sched['poly']['initial'],max_epochs=self.p.k_r_sched['poly']['max_epochs'],poly_power=self.p.k_r_sched['poly']['poly_power'])

        lr_U = list(self.p.k_U_sched.keys())[0]
        if lr_U == 'constant':
            self.k_U_lr = partial(constant_lr,initial=self.p.k_U_sched['constant']['initial'])
        elif lr_U == 'step':
            self.k_U_lr = partial(step_decay_lr,initial=self.p.k_U_sched['step']['initial'],drop_every=self.p.k_U_sched['step']['drop_every'],drop_factor=self.p.k_U_sched['step']['drop_factor'])
        elif lr_U == 'poly':
            self.k_U_lr = partial(polynomial_decay_lr,initial=self.p.k_U_sched['poly']['initial'],max_epochs=self.p.k_U_sched['poly']['max_epochs'],poly_power=self.p.k_U_sched['poly']['poly_power'])

        lr_o = list(self.p.k_o_sched.keys())[0]
        if lr_o == 'constant':
            self.k_o_lr = partial(constant_lr,initial=self.p.k_o_sched['constant']['initial'])
        elif lr_o == 'step':
            self.k_o_lr = partial(step_decay_lr,initial=self.p.k_o_sched['step']['initial'],drop_every=self.p.k_o_sched['step']['drop_every'],drop_factor=self.p.k_o_sched['step']['drop_factor'])
        elif lr_o == 'poly':
            self.k_o_lr = partial(polynomial_decay_lr,initial=self.p.k_o_sched['poly']['initial'],max_epochs=self.p.k_o_sched['poly']['max_epochs'],poly_power=self.p.k_o_sched['poly']['poly_power'])


    def train(self, X, Y):
        """
        X: input matrix (two cases)
            i. num_imgs x numxpxls x numypxls: set of whole images
            ii. numtiles x numtlxpxls x numtlypxls: set of tiles
                NOTE: these cases will only be distinguishable using PCC attribute self.num_r1_mods (if > 1: tiled, if == 1, non-tiled)

        Model layer architecture not initialized until model.train() called (train() needs to read X for tile dimensions)
        """

        ## Detect WHOLE IMAGE case: model will be constructed with only 1 r[1] module
        if self.p.num_r1_mods == 1:

            # Set attrs
            self.is_tiled = False
            self.sgl_tile_area = 0

            ## Initiate r[0] - r[n] layers, U[1] - U[n] layers

            # Input IMAGE layer of size (self.p.input_size,1)
            self.r[0] = np.random.randn(self.p.input_size,1)

            # Non-input layers (1 - n): hidden layers (1 - n-1) plus output layer (Li's "localist" layer n)
            # Ought maybe to switch initialized r, U's from strictly Gaussian (randn) to tunable based on specified model.p.r_prior, U_prior
            for layer_num in range(1, self.num_nonin_lyrs + 1)
                self.r[layer_num] = np.random.randn(self.p.hidden_sizes[i-1],1)
                self.U[layer_num] = np.random.randn(len(self.r[i-1]),len(self.r[i]))


        ## Detect TILED image case: model will be constructed with num r[1] modules (num_r1_mods) == (dataset) numtiles [specified in main.py]
        elif self.p.num_r1_mods > 1:

            # Set attrs
            self.is_tiled = True
            # X dims for Li case should be: (5, 225, 256)
            self.sgl_tile_area = X.shape[2]

            ## Initiate r[0] - r[n] layers, U[1] - U[n] layers; tiled case

            # Input image TILES layer of size (number of tiles == number of r1 modules, area of a single tile); Li case: 225, 256
            self.r[0] = np.random.randn(self.p.num_r1_mods,self.sgl_tile_area)

            # Non-input layers (1 - n): hidden layers (1 - n-1) plus output layer (Li's "localist" layer n)
            # Ought maybe to switch initialized r, U's from strictly Gaussian (randn) to tunable based on specified model.p.r_prior, U_prior

            ## Hidden layer 1 & 2 first (only hidden layers directly dependent on num tiles)
            # Hidden layer 1
            # Li case: r1: 225, 32
            #          U1: 225, 256, 32
            self.r[1] = np.random.randn(self.p.num_r1_mods, self.p.hidden_sizes[0])
            self.U[1] = np.random.randn(self.p.num_r1_mods, self.sgl_tile_area, self.p.hidden_sizes[0])

            # Hidden layer 2
            # Li case: r2: 128,
            #          U2: 7200 (225*32), 128
            self.r[2] = np.random.randn(self.p.hidden_sizes[1])
            self.U[2] = np.random.randn(self.p.num_r1_mods * self.p.hidden_sizes[0], self.p.hidden_sizes[1])

            # Hidden layers > 2
            for layer_num in range(1, self.num_nonin_lyrs + 1)
                self.r[layer_num] = np.random.randn(self.p.hidden_sizes[i-1],1)
                self.U[layer_num] = np.random.randn(len(self.r[i-1]),len(self.r[i]))

            # "Localist" layer

        else:
            print("Model.num_r1_mods attribute needs to be in [1,n=int<<inf]")
            exit()


        # If self.p.class_scheme == 'c2', initiate o and Uo layers
        if self.p.class_scheme == 'c2':
            # Initialize output layer (Li case: 5, 1)
            self.o = np.random.randn(self.p.output_size,1)
            # And final set of weights to the output (Li case: 5, 128)
            self.U_o = np.random.randn(self.p.output_size, self.p.hidden_sizes[-1])


        ### Non-tiled (whole) image input case

        if self.is_tiled is False:
            print("Model training on NON-TILED input")
            exit()

        ### Tiled (sliced) image input case

        elif self.is_tiled is True:
                print("Model training on TILED input")
                exit()
