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

def model_find_and_or_create(num_nonin_lyrs=3, lyr_sizes=(96,128,5), num_r1_mods=225, act_fxn="lin",
    r_prior="kurt", U_prior="kurt", class_scheme="c1", num_epochs=500):

    ### Directory search for named model

    # Initiate model name string
    desired_model = "mod.{}_".format(num_nonin_lyrs)

    if len(lyr_sizes) != num_nonin_lyrs:
        print("Number of non-input layers (num_nonin_lyrs) must == length of lyr_sizes tuple")
        exit()

    for lyr in range(0,num_nonin_lyrs):
        str_lyr = str(lyr_sizes[lyr])
        if lyr < num_nonin_lyrs - 1:
            desired_model += (str_lyr + "-")
        else:
            desired_model += (str_lyr + "_")

    ### Check for model in local directory: if present, quit (creation / training not needed); if not, create, save

    desired_model += "{}_{}_{}_{}_{}_{}.pydb".format(num_r1_mods, act_fxn, r_prior, U_prior, class_scheme, num_epochs)

    print("II. Desired model is {}".format(desired_model) + "\n")

    if os.path.exists("./" + desired_model):
        print("Desired model " + desired_model + " already present in local dir: would you like to overwrite it? (y/n)")
        ans = input()
        # For overwrite
        if ans == "y":
            # Initialize model
            mod = PredictiveCodingClassifier(p)

        else:
            print("Quitting main.py..." + "\n")
            exit()
    # For first save
    else:
        # Initialize model
        mod = PredictiveCodingClassifier(p)

    return mod


class PredictiveCodingClassifier:
    def __init__(self, parameters):

        self.p = parameters

        # Choices for transformation functions, priors
        self.act_fxn_dict = {'lin':linear_trans,'tan':tanh_trans}
        self.prior_dict = {'gaus':gauss_prior, 'kurt':kurt_prior}

        # all the representations (including the image r[0] which is not trained)
        self.r = {}
        # synaptic weights controlling reconstruction in the network
        self.U = {}

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




    def train(self, X, y):
        """
        X: input matrix (two cases)
            i. num_imgs x numxpxls x numypxls: set of whole images
            ii. numtiles x numtlxpxls x numtlypxls: set of tiles
                NOTE: these cases will only be distinguishable using PCC attribute self.num_r1_mods (if < 1: tiled, if == 1, non-tiled)
        """

        ### Non-tiled (whole) image input case

        if self.p.num_r1_mods == 1:
            print("Model training on NON-TILED input")
            exit()

        ### Tiled (sliced) image input case

        elif self.p.num_r1_mods > 1:
            print("Model training on TILED input")
            exit()

        else:
            print("Model num_r1_mods attribute needs to be in [1,n=int<inf]")
            exit()
