import numpy as np
from matplotlib import pyplot as plt
from learning import *
from functools import partial
import math
import data
import cv2
from parameters import ModelParameters
from sys import exit
import pickle
import time
import datetime

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
        self.num_tot_lyrs = self.num_nonin_lyrs + 1

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
            i. num_imgs, numxpxls * numypxls: set of whole images
            ii. num_imgs * numtiles, numtlxpxls * numtlypxls: set of tiles
                NOTE: these cases will only be distinguishable using PCC attribute self.num_r1_mods (if > 1: tiled, if == 1, non-tiled)

        Model layer architecture not initialized until model.train() called (train() needs to read X for tile dimensions)
        """

        #### ARCHITECTURE INITIALIZATION

        ## Detect WHOLE IMAGE case: model will be constructed with only 1 r[1] module
        if self.p.num_r1_mods == 1:

            # Set attrs
            self.is_tiled = False
            self.sgl_tile_area = 0

            ## Initiate r[0] - r[n] layers, U[1] - U[n] layers
            # Li uses np.zeros for r inits, np.random.rand (unif) for U inits: may have to switch to those

            # Input IMAGE layer of size (self.p.input_size,1)
            self.r[0] = np.random.randn(self.p.input_size, 1)

            # Non-input layers (1 - n): hidden layers (1 - n-1) plus output layer (Li's "localist" layer n)
            # Ought maybe to switch initialized r, U's from strictly Gaussian (randn) to tunable based on specified model.p.r_prior, U_prior
            # Hidden layers
            for layer_num in range(1, self.num_nonin_lyrs + 1):
                self.r[layer_num] = np.random.randn(self.p.hidden_sizes[i-1], 1)
                self.U[layer_num] = np.random.randn(len(self.r[i-1]), len(self.r[i]))

            # "Localist" layer (relates size of Y (num classes) to final hidden layer)
            self.r[self.num_nonin_lyrs] = np.random.randn(self.p.output_size, 1)
            self.U[self.num_nonin_lyrs] = np.random.randn(len(self.r[i-1]), len(self.r[i]))


        ## Detect TILED image case: model will be constructed with num r[1] modules (num_r1_mods) == (dataset) numtiles [specified in main.py]
        elif self.p.num_r1_mods > 1:

            # Set attrs
            self.is_tiled = True
            # X dims for Li case should be: (1125, 256); thus sgl_tile_area 256
            self.sgl_tile_area = X.shape[1]

            ## Initiate r[0] - r[n] layers, U[1] - U[n] layers; tiled case
            # Li uses np.zeros for r inits, np.random.rand (unif) for U inits: may have to switch to those

            # Input image TILES layer of size (number of tiles == number of r1 modules, area of a single tile); Li case: 225, 256
            self.r[0] = np.random.randn(self.p.num_r1_mods,self.sgl_tile_area)

            # Non-input layers (1 - n): hidden layers (1 - n-1) plus output layer (Li's "localist" layer n)
            # Ought maybe to switch initialized r, U's from strictly Gaussian (randn) to tunable based on specified model.p.r_prior, U_prior

            ## Hidden layers 1 & 2 first (only hidden layers directly dependent on num tiles)
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

            if self.num_hidden_lyrs > 2:
                # Hidden layer 3 or more
                for layer_num in range(3, self.num_nonin_lyrs + 1):
                    self.r[layer_num] = np.random.randn(self.p.hidden_sizes[layer_num-1], 1)
                    self.U[layer_num] = np.random.randn(len(self.r[layer_num-1]), len(self.r[layer_num]))

            # "Localist" layer (relates size of Y (num classes) to final hidden layer)
            self.r[self.num_nonin_lyrs] = np.random.randn(self.p.output_size, 1)
            self.U[self.num_nonin_lyrs] = np.random.randn(len(self.r[self.num_nonin_lyrs-1]), len(self.r[self.num_nonin_lyrs]))

        else:
            print("Model.num_r1_mods attribute needs to be in [1,n=int<<inf]")
            exit()

        # If self.p.class_scheme == 'c2', initiate o and Uo layers
        # NOTE: May have to change these sizes to account for Li localist layer (is o redundant in that case?)
        if self.p.class_scheme == 'c2':
            # Initialize output layer (Li case: 5, 1)
            self.o = np.random.randn(self.p.output_size,1)
            # And final set of weights to the output (Li case: 5, 128)
            self.U_o = np.random.randn(self.p.output_size, self.p.hidden_sizes[-1])


        #### MODEL PARAM READOUT IN TERMINAL

        print("Model parameters by layer:")

        tot_num_params = 0
        # Save for metadata too
        rs = {}
        Us = {}

        for r, rvec in self.r.items():
            rdims = []
            for rdim in rvec.shape:
                rdims.append(rdim)

            rshape = ""
            rparams = 1
            for rdim in rdims:
                rshape += str(rdim) + ","
                rparams *= rdim

            print(f"r[{r}] size: ({rshape}); total params: {rparams}")
            tot_num_params += rparams

        for U, Uvec in self.U.items():
            Udims = []
            for Udim in Uvec.shape:
                Udims.append(Udim)

            Ushape = ""
            Uparams = 1
            for Udim in Udims:
                Ushape += str(Udim) + ","
                Uparams *= Udim

            print(f"U[{U}] size: ({Ushape}); total params: {Uparams}")
            tot_num_params += Uparams

        # If C2 classification: o and Uo layers
        if self.p.class_scheme == 'c2':
            odims = []
            for odim in self.o.shape:
                odims.append(odim)

            oshape = ""
            oparams = 1
            for odim in odims:
                oshape += str(odim) + ","
                oparams *= odim
            print(f"o size: ({oshape}); total params: {oparams}")
            tot_num_params += oparams

            Uodims = []
            for Uodim in self.U_o.shape:
                Uodims.append(Uodim)

            Uoshape = ""
            Uoparams = 1
            for Uodim in Uodims:
                Uoshape += str(Uodim) + ","
                Uoparams *= Uodim
            print(f"Uo size: ({Uoshape}); total params: {Uoparams}")
            tot_num_params += Uoparams

        # Take out size of input (these are not actually parameters within the model proper)

        if self.is_tiled is False:
            # Whole image case: r0 == self.p.input_size
            tot_num_params -= self.p.input_size

        elif self.is_tiled is True:
            # Tiled image case: r0 != self.p.input_size
            tot_num_params -= self.r[0].shape[0] * self.r[0].shape[1]

        # Print total num model params
        print(f"Total number of model parameters: {tot_num_params}" + "\n")

        #### PICKLE UNTRAINED MODEL (EPOCH "0") AND METADATA

        model_name = f"mod.{self.num_nonin_lyrs}_"

        # Get layer sizes for (hl1-hl2-...hln-localist) model pickle naming format
        lyr_sizes = []
        for nonin_lyr in range(0, self.num_hidden_lyrs):
            lyr_sizes.append(self.p.hidden_sizes[nonin_lyr])
        lyr_sizes.append(self.p.output_size)

        # Add those sizes to the model name string
        for lyr in range(0,self.num_nonin_lyrs):
            str_lyr = str(lyr_sizes[lyr])
            if lyr < self.num_nonin_lyrs - 1:
                model_name += (str_lyr + "-")
            else:
                model_name += (str_lyr + "_")

        # Add the rest of the model naming params to name
        model_name += f"{self.p.num_r1_mods}_{self.p.act_fxn}_{self.p.r_prior}_{self.p.U_prior}_{self.p.class_scheme}_"
        model_name_pre_epoch = model_name
        model_name_untrained = model_name_pre_epoch + "0"
        model_pkl_name = model_name_untrained + ".pydb"
        model_metadata_name = model_name_untrained + ".txt"

        print(f"Untrained model name is {model_name}")

        # Pickle model
        with open(model_pkl_name, "wb") as model_out:
            pickle.dump(self, model_out)
            print(f"Untrained model {model_pkl_name} saved in local dir")

        ### METADATA (model attributes that aren't listed in model name); e.g. learning rate at epoch

        header = f"Metadata for UNTRAINED model {model_pkl_name} \n"
        is_tiled = f"Tiled: {self.is_tiled}"
        update_scheme = f"Update scheme: {self.p.update_scheme}"
        batch_size = f"Batch size: {self.p.batch_size}"
        epoch_counter = f"Number of epochs completed / Total number of epochs in regimen: 0 / {self.p.num_epochs}"
        k_r_sched = f"r LR schedule: {self.p.k_r_sched}"
        k_r_at_start = f"r LR at start (has not occurred yet): {self.k_r_lr(0)}"
        k_U_sched = f"U LR schedule: {self.p.k_U_sched}"
        k_U_at_start = f"U LR at start (has not occurred yet): {self.k_U_lr(0)}"
        k_o_sched = f"o LR schedule: {self.p.k_o_sched}"
        k_o_at_start = f"o LR at start (has not occurred yet): {self.k_o_lr(0)}"
        sigma_sq = f"Sigma squared values at each layer: {self.p.sigma_sq}"
        alpha = f"Alpha values at each layer: {self.p.alpha}"
        lam = f"Lambda values at each layer: {self.p.lam}"
        size_of_starting_img = f"Num params in an original whole input image, regardless of whether images will become tiled for training: {self.p.input_size}"
        time_created = datetime.datetime.now()
        time_at_chkpt = time_created
        train_time_elapsed = time_at_chkpt - time_created
        time_created_str = f"Time at model creation: {time_created}"
        time_at_chkpt = f"Time at checkpoint: {time_at_chkpt}"
        train_time_elapsed = f"Training time elapsed: {train_time_elapsed}"

        metadata_lines = [header, is_tiled, update_scheme, batch_size, epoch_counter, k_r_sched,
                            k_r_at_start, k_U_sched, k_U_at_start, k_o_sched, k_o_at_start, sigma_sq, alpha, lam, size_of_starting_img,
                            time_created_str, time_at_chkpt, train_time_elapsed]

        # Write metadata
        with open(model_metadata_name, "w") as metadata_out:
            for line in metadata_lines:
                metadata_out.write(line)
                metadata_out.write("\n")
            print(f"Untrained model metadata {model_metadata_name} saved in local dir" + "\n")


        #### TRAINING LOGIC

        ### Parse which epochs to checkpoint

        chkpt_scheme = self.p.checkpointing[0]

        if chkpt_scheme == "fraction":

            divisor = self.p.checkpointing[1]
            chkpt_every_n = int(self.p.num_epochs / divisor)

        elif chkpt_scheme == "every_n_ep":

            chkpt_every_n = int(self.p.checkpointing[1])

        elif chkpt_scheme == "off":

            chkpt_every_n = self.p.num_epochs

        ### WHOLE IMAGE TRAINING case
        if self.is_tiled is False:
            print("Model training on NON-TILED input")
            print("non-tiled not yet written, quitting...")
            exit()

        ### TILED TRAINING case
        elif self.is_tiled is True:
            print("Model training on TILED input")
            print(f"Area of a single tile is: {self.sgl_tile_area}" + "\n")

            for epoch in range(1, self.p.num_epochs + 1):

                print(f"Epoch: {epoch}")

                # Checkpointing logic
                if epoch % chkpt_every_n == 0:

                    # Pickle model
                    mod_chkpt_name = model_name_pre_epoch + str(epoch)
                    mod_chkpt_name_pkl = mod_chkpt_name + ".pydb"

                    with open(mod_chkpt_name_pkl, "wb") as model_out:
                        pickle.dump(self, model_out)
                        print(f"Trained model at epoch {epoch} {mod_chkpt_name_pkl} saved in local dir")

                    # Save current metadata
                    header = f"Metadata for TRAINED model at epoch {epoch} {mod_chkpt_name_pkl} \n"
                    is_tiled = f"Tiled: {self.is_tiled}"
                    update_scheme = f"Update scheme: {self.p.update_scheme}"
                    batch_size = f"Batch size: {self.p.batch_size}"
                    epoch_counter = f"Number of epochs completed / Total number of epochs in regimen: {epoch} / {self.p.num_epochs}"
                    k_r_sched = f"r LR schedule: {self.p.k_r_sched}"
                    k_r_at_start = f"r LR at start of ep {epoch}: {self.k_r_lr(epoch)}"
                    k_U_sched = f"U LR schedule: {self.p.k_U_sched}"
                    k_U_at_start = f"U LR at start of ep {epoch}: {self.k_U_lr(epoch)}"
                    k_o_sched = f"o LR schedule: {self.p.k_o_sched}"
                    k_o_at_start = f"o LR at start of ep {epoch}: {self.k_o_lr(epoch)}"
                    sigma_sq = f"Sigma squared values at each layer: {self.p.sigma_sq}"
                    alpha = f"Alpha values at each layer: {self.p.alpha}"
                    lam = f"Lambda values at each layer: {self.p.lam}"
                    size_of_starting_img = f"Num params in an original whole input image, regardless of whether images will become tiled for training: {self.p.input_size}"
                    time_created = time_created
                    time_at_chkpt = datetime.datetime.now()
                    train_time_elapsed = time_at_chkpt - time_created
                    time_created_str = f"Time at model creation: {time_created}"
                    time_at_chkpt = f"Time at checkpoint: {time_at_chkpt}"
                    train_time_elapsed = f"Training time elapsed at end of ep {epoch}: {train_time_elapsed}"

                    metadata_lines = [header, is_tiled, update_scheme, batch_size, epoch_counter, k_r_sched,
                                        k_r_at_start, k_U_sched, k_U_at_start, k_o_sched, k_o_at_start, sigma_sq, alpha, lam, size_of_starting_img,
                                        time_created_str, time_at_chkpt, train_time_elapsed]

                    # Write metadata
                    mod_chkpt_name_txt = mod_chkpt_name + ".txt"
                    with open(mod_chkpt_name_txt, "w") as metadata_out:
                        for line in metadata_lines:
                            metadata_out.write(line)
                            metadata_out.write("\n")
                        print(f"Trained model metadata at epoch {epoch} {mod_chkpt_name_txt} saved in local dir" + "\n")
