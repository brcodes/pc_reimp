import numpy as np
import matplotlib as mpl
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
        lr_r_sched = list(self.p.k_r_sched.keys())[0]
        if lr_r_sched == 'constant':
            self.k_r_lr = partial(constant_lr,initial=self.p.k_r_sched['constant']['initial'])
        elif lr_r_sched == 'step':
            self.k_r_lr = partial(step_decay_lr,initial=self.p.k_r_sched['step']['initial'],drop_every=self.p.k_r_sched['step']['drop_every'],drop_factor=self.p.k_r_sched['step']['drop_factor'])
        elif lr_r_sched == 'poly':
            self.k_r_lr = partial(polynomial_decay_lr,initial=self.p.k_r_sched['poly']['initial'],max_epochs=self.p.k_r_sched['poly']['max_epochs'],poly_power=self.p.k_r_sched['poly']['poly_power'])

        lr_U_sched = list(self.p.k_U_sched.keys())[0]
        if lr_U_sched == 'constant':
            self.k_U_lr = partial(constant_lr,initial=self.p.k_U_sched['constant']['initial'])
        elif lr_U_sched == 'step':
            self.k_U_lr = partial(step_decay_lr,initial=self.p.k_U_sched['step']['initial'],drop_every=self.p.k_U_sched['step']['drop_every'],drop_factor=self.p.k_U_sched['step']['drop_factor'])
        elif lr_U_sched == 'poly':
            self.k_U_lr = partial(polynomial_decay_lr,initial=self.p.k_U_sched['poly']['initial'],max_epochs=self.p.k_U_sched['poly']['max_epochs'],poly_power=self.p.k_U_sched['poly']['poly_power'])

        lr_o_sched = list(self.p.k_o_sched.keys())[0]
        if lr_o_sched == 'constant':
            self.k_o_lr = partial(constant_lr,initial=self.p.k_o_sched['constant']['initial'])
        elif lr_o_sched == 'step':
            self.k_o_lr = partial(step_decay_lr,initial=self.p.k_o_sched['step']['initial'],drop_every=self.p.k_o_sched['step']['drop_every'],drop_factor=self.p.k_o_sched['step']['drop_factor'])
        elif lr_o_sched == 'poly':
            self.k_o_lr = partial(polynomial_decay_lr,initial=self.p.k_o_sched['poly']['initial'],max_epochs=self.p.k_o_sched['poly']['max_epochs'],poly_power=self.p.k_o_sched['poly']['poly_power'])

        # Avg cost per epoch during training; just representation terms
        self.avg_E_per_ep = []
        # Avg cost per epoch during training; just classification terms
        self.avg_C_per_ep = []
        # Avg prediction error across all layers, avg'd over each image in epoch, during training
        self.avg_PE_all_lyrs_avg_per_ep = []
        # Accuracy per epoch during training
        self.acc_per_ep = []

    def rep_cost(self):
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        '''
        E = 0
        # LSQ cost
        PE_list = []
        for i in range(0,len(self.r)-1):
            v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
            vTdotv = v.T.dot(v)
            E = E + ((1 / self.p.sigma_sq[i+1]) * vTdotv)[0,0]

            # Also calulate prediction error for each layer
            PE = np.sqrt(vTdotv)
            PE_list.append(PE)

        # priors on r[1],...,r[n]; U[1],...,U[n]
        for i in range(1,len(self.r)):
            E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])

        return (E, PE_list)

    def class_cost_nc(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method NC (always 0 with NC: no label data used to train).
        Also, guesses image: returns 0/1 if max arg of softmax(r[n]) doesn't/does match label one-hot elem. """

        # Cost
        NC = 0

        # Guess image
        sm_rn = softmax(self.r[n])

        if np.argmax(sm_rn) == np.argmax(label[:,None]):
            guess_correct_or_not = 1
            self.n_correct_classifs_per_ep += 1
        else:
            guess_correct_or_not = 0

        return NC, guess_correct_or_not

    def class_cost_c1(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C1. Also, guesses image: returns 0/1 if max arg
        of softmax(r[n]) doesn't/does match label one-hot elem. """

        n = self.num_hidden_lyrs
        sm_rn = softmax(self.r[n])

        # Calc cost
        C1 = -(1/2)*(label[None,:] - sm_rn).T.dot(label[None,:] - sm_rn)[0,0]

        # Guess image
        if np.argmax(sm_rn) == np.argmax(label[:,None]):
            guess_correct_or_not = 1
            self.n_correct_classifs_per_ep += 1
        else:
            guess_correct_or_not = 0

        return C1, guess_correct_or_not

    def class_cost_c2(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C2. Also, guesses image: returns 0/1 if max arg
        of softmax(r[n]) doesn't/does match label one-hot elem. """

        n = self.num_hidden_lyrs

        # Calc cost
        C2 = -1*label[None,:].dot(np.log(softmax((self.U_o.dot(self.r[n])))))[0,0]

        # Guess image
        if np.argmax(softmax(self.r[n])) == np.argmax(label[:,None]):
            guess_correct_or_not = 1
            self.n_correct_classifs_per_ep += 1
        else:
            guess_correct_or_not = 0

        return C2, guess_correct_or_not

    def train(self, X, Y):
        """
        X: input matrix (two cases)
            i. num_imgs, numxpxls * numypxls: set of whole images
            ii. num_imgs * numtiles, numtlxpxls * numtlypxls: set of tiles
                NOTE: these cases will only be distinguishable using PCC attribute self.num_r1_mods (if > 1: tiled, if == 1, non-tiled)

        Model layer architecture not initialized until model.train() called (train() needs to read X for tile dimensions)
        """

        print(f"beginning X.shape is {X.shape}")
        print(f"beginning Y.shape is {Y.shape}")

        #### ARCHITECTURE INITIALIZATION

        ## Detect WHOLE IMAGE case: model will be constructed with only 1 r[1] module
        if self.p.num_r1_mods == 1:

            # Set some general attrs
            self.is_tiled = False
            self.sgl_tile_area = 0
            self.num_training_imgs = X.shape[0]

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

            # Set some general attrs
            self.is_tiled = True
            # X dims for Li case should be: (1125, 256); thus sgl_tile_area 256
            self.sgl_tile_area = X.shape[1]
            self.num_tiles_per_img = self.p.num_r1_mods
            self.num_training_imgs = int(X.shape[0] / self.num_tiles_per_img)

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
        r_param_lines = []
        U_param_lines = []

        for r, rvec in self.r.items():
            rdims = []
            for rdim in rvec.shape:
                rdims.append(rdim)

            rshape = ""
            rparams = 1
            for rdim in rdims:
                rshape += str(rdim) + ","
                rparams *= rdim

            # Print and save for metadata too
            r_param_line = f"r[{r}] size: ({rshape}); total params: {rparams}"
            r_param_lines.append(r_param_line)
            print(r_param_line)

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

            # Print and save for metadata
            U_param_line = f"U[{U}] size: ({Ushape}); total params: {Uparams}"
            U_param_lines.append(U_param_line)
            print(U_param_line)

            tot_num_params += Uparams

        # Dummy line variables for non-C2 cases
        o_param_line = "o size: n/a"
        Uo_param_line = "Uo size: n/a"

        # If C2 classification: o and Uo layers
        if self.p.class_scheme == 'c2':
            o = []
            Uo = []

            odims = []
            for odim in self.o.shape:
                odims.append(odim)

            oshape = ""
            oparams = 1
            for odim in odims:
                oshape += str(odim) + ","
                oparams *= odim

            # Print and save for metadata
            o_param_line = f"o size: ({oshape}); total params: {oparams}"
            print(o_param_line)

            tot_num_params += oparams

            Uodims = []
            for Uodim in self.U_o.shape:
                Uodims.append(Uodim)

            Uoshape = ""
            Uoparams = 1
            for Uodim in Uodims:
                Uoshape += str(Uodim) + ","
                Uoparams *= Uodim

            # Print and save for metadata
            Uo_param_line = f"Uo size: ({Uoshape}); total params: {Uoparams}"
            print(Uo_param_line)

            tot_num_params += Uoparams

        # Take out size of input (these are not actually parameters within the model proper)

        if self.is_tiled is False:
            # Whole image case: r0 == self.p.input_size
            tot_num_params -= self.p.input_size

        elif self.is_tiled is True:
            # Tiled image case: r0 != self.p.input_size
            tot_num_params -= self.r[0].shape[0] * self.r[0].shape[1]

        # Print and save for metadata
        tot_params_line = f"Total number of model parameters: {tot_num_params}" + "\n"
        print(tot_params_line)

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

        print(f"Untrained model name is {model_pkl_name}")

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
        k_r_at_start = f"r LR at start of ep 1 (has not occurred yet): {self.k_r_lr(0)}"
        k_U_sched = f"U LR schedule: {self.p.k_U_sched}"
        k_U_at_start = f"U LR at start of ep 1 (has not occurred yet): {self.k_U_lr(0)}"
        k_o_sched = f"o LR schedule: {self.p.k_o_sched}"
        k_o_at_start = f"o LR at start of ep 1 (has not occurred yet): {self.k_o_lr(0)}"
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

        # Write metadata untrained model
        with open(model_metadata_name, "w") as metadata_out:
            for line in metadata_lines:
                metadata_out.write(line)
                metadata_out.write("\n")

            metadata_out.write("\n")
            for line in r_param_lines:
                metadata_out.write(line)
                metadata_out.write("\n")

            for line in U_param_lines:
                metadata_out.write(line)
                metadata_out.write("\n")

            metadata_out.write(o_param_line)
            metadata_out.write("\n")
            metadata_out.write(Uo_param_line)
            metadata_out.write("\n")
            metadata_out.write(tot_params_line)
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

        ### Set appropriate classification cost function
        class_costs_dict = {"nc": self.class_cost_nc, "c1": self.class_cost_c1, "c2": self.class_cost_c2}

        if self.p.class_scheme == "c1":
            class_cost = class_costs_dict["c1"]

        elif self.p.class_scheme == "c2":
            class_cost = class_costs_dict["c2"]

        elif self.p.class_scheme == "nc":
            class_cost = class_costs_dict["nc"]


        ### WHOLE IMAGE TRAINING case
        if self.is_tiled is False:
            print("Model training on NON-TILED input")
            print("non-tiled not yet written")

            if self.update_scheme == "rU_simultaneous":
                print("non-tiled rU simultaneous not yet written, quitting...")

            elif self.update_scheme == "r_eq_then_U":
                print("non-tiled r eq then U not yet written, quitting...")

            exit()

        ### TILED TRAINING case
        elif self.is_tiled is True:
            print("Model training on TILED input")
            print(f"Area of a single tile is: {self.sgl_tile_area}" + "\n")

            # Split X (alltiles, tilearea) into (num imgs, tiles for one img, tilearea)
            tlsidxlo = 0
            tlsidxhi = self.num_tiles_per_img

            split_X = []

            # NOTE: If interested in speed later, modify PCC program such that data.py saves X in form (numimgs, num_tiles_per_img, tilearea)
            # and train() takes it in that format as well.
            # This indexing and splitting operation will add a lengthy preprocessing step to the beginning of train() if numimgs >>~ 1000
            for image in range(0, self.num_training_imgs):
                split_single = X[tlsidxlo:tlsidxhi]
                split_X.append(split_single)
                tlsidxlo += self.num_tiles_per_img
                tlsidxhi += self.num_tiles_per_img

            split_X = np.array(split_X)
            # print(f"split_X.shape is {split_X.shape}")
            X = split_X
            # print(f"later X.shape is {X.shape}")

            ### Li case: and updating proceeds through layers, r and U of a layer i are updated simultaneously 30 times
            ### This means each each layer's r/U updates 30 times per image, using top down and bottom up information specific to that image
            if self.update_scheme == "rU_simultaneous":
                print("rU simultaneous TRAINING about to begin")

                for epoch in range(1, self.p.num_epochs + 1):

                    print(f"Epoch: {epoch}")

                    # Shuffle indices of X, Y together, each epoch
                    N_permuted_indices = np.random.permutation(self.num_training_imgs)
                    X_shuffled = X[N_permuted_indices]
                    print(f"X_shuffled.shape is {X_shuffled.shape}")
                    Y_shuffled = Y[N_permuted_indices]

                    ### Initialize epoch-dependent measures
                    # Representation cost reset
                    E_over_all_imgs = 0
                    # Classification cost reset
                    C_over_all_imgs = 0
                    # Prediction error reset
                    PEs_over_all_imgs = []
                    # Number of correct classifications
                    self.n_correct_classifs_per_ep = 0

                    # Set learning rates at the start of each epoch
                    k_r = self.k_r_lr(epoch-1)
                    k_U = self.k_U_lr(epoch-1)
                    k_o = self.k_o_lr(epoch-1)

                    """
                    # NOTE: for internal plotting; remove or wrap later
                    tiles_of_all_images = []
                    imgidxlo = 0
                    imgidxhi = self.num_tiles_per_img
                    # RB99 tiling scheme case
                    if self.num_tiles_per_img == 3:
                        tilecols = 3
                        tilerows = 1

                    # RB97a and Li tiling scheme cases
                    elif self.num_tiles_per_img == 4 or self.num_tiles_per_img == 225:
                        tilecols = int(np.sqrt(self.num_tiles_per_img))
                        tilerows = tilecols
                    # NOTE: to above # NOTE
                    """

                    #### GRADIENT DESCENT LOOP
                    for image in range(0, self.num_training_imgs):

                        ## Set image and label
                        # "Sgl image" is really a set of n (self.num_tiles_per_img) tiles; Li case: shape (225, 256)
                        single_image = X_shuffled[image]
                        print(f"single_image.shape is {single_image.shape}")
                        label = Y_shuffled[image]
                        print(f"single label.shape image {image+1} is {label.shape}")


                        """
                        INTERNAL PLOTTING TEST FOR SHUFFLING: WRAP OR REMOVE LATER
                        """

                        """
                        # NOTE: for internal plotting; remove or wrap later
                        tiles_of_single_img = []
                        tl_num = 1

                        for tl in single_image:

                            # Expand tiles for plotting
                            expanded_tile = data.inflate_vectors(tl[None,:], (16,16))
                            expanded_tile = np.squeeze(expanded_tile)

                            tiles_of_single_img.append(expanded_tile)

                            # Single tile plotting
                            # tl = np.array(expanded_tile).astype(float)
                            # plt.imshow(tl, cmap="gray")
                            # plt.title("{}".format(desired_dataset) + "\n" + "image {} ".format(img+1) + "tile {}".format(tl_num))
                            # plt.show()

                            tl_num += 1

                        tiles_of_all_images.append(tiles_of_single_img)

                        imgidxlo += self.num_tiles_per_img
                        imgidxhi += self.num_tiles_per_img

                        # Tile stacking and plotting (stacked collage) loop
                        tiles_of_all_images = np.array(tiles_of_all_images, dtype=list)

                        print(f"Shape of array with all tiles parsed by image: {tiles_of_all_images.shape}")

                        # In Li case, this is (225,16,16) but generally is (numtiles, tlxpxls, tlypxls)
                        tiles_of_single_img = tiles_of_all_images[image]

                        print(f"Shape of array with tiles for image {image}: {tiles_of_single_img.shape}")

                        rowidxlo = 0
                        rowidxhi = tilecols

                        vstackedrows = np.zeros([16,16*tilecols])

                        for row in range(0,tilerows):

                            onerow = tiles_of_single_img[rowidxlo:rowidxhi,:,:]

                            # Initiate left-to-right stacking, completing a full row
                            hstackedrow = np.array(onerow[0])[:,:]

                            for col in range(1, tilecols):
                                nonfirsttileinrow = np.array(onerow[col])[:,:]
                                hstackedrow = np.concatenate((hstackedrow, nonfirsttileinrow), axis=1)

                            # With collaged row complete, stack rows for full collage
                            vstackedrows = np.vstack([vstackedrows, hstackedrow])

                            rowidxlo += tilecols
                            rowidxhi += tilecols

                        # Remove collage row of zeros used to initiate collage, convert to float64
                        stackedtilecollage = vstackedrows[16:,:].astype(float)

                        # Plot
                        plt.imshow(stackedtilecollage, cmap="gray")
                        plt.title(f"mod.train(X) input verif. epoch {epoch}: Li dataset img {image+1}" + "\n" +
                                "{}-tile collage ({}x{}tls)".format(self.num_tiles_per_img, tilerows, tilecols))

                        plt.show()
                        tiles_of_all_images = tiles_of_all_images.tolist()
                        print(f" type of tiles of all images {type(tiles_of_all_images)}")
                        # NOTE: ABOVE TO OTHER NOTE is for internal plotting; remove or wrap later
                        """


                        for i in range(1, n):

                            # r update
                            self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                            * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                            + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                            - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

                            # U update
                            self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                            * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                            - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]

                        self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                        * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                        - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

                        # Add classification cost component of final representation update (changes based on C1, C2 or NC setting)
                        + ((k_o) * (label[:,None] - softmax(self.r[n])))

                        # U[n] update (C1, C2) (identical to U[i], except index numbers)
                        self.U[n] = self.U[n] + (k_U / self.p.sigma_sq[n]) \
                        * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                        - (k_U / 2) * self.h(self.U[n],self.p.lam[n])[1]

                        # Training loss function E and PE by layer
                        Eimg, PE_list_img = self.rep_cost()

                        # Add
                        E = E + Eimg

                        # PE for each layer

                        # Classification loss function C (calls relevant fxn: C1, C2, or zero multiplier for NC)
                        C, guess_correct_or_not = class_cost(label)
                        E = E + C










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
                        k_r_at_start = f"r LR for ep {epoch}: {self.k_r_lr(epoch-1)}"
                        k_U_sched = f"U LR schedule: {self.p.k_U_sched}"
                        k_U_at_start = f"U LR for ep {epoch}: {self.k_U_lr(epoch-1)}"
                        k_o_sched = f"o LR schedule: {self.p.k_o_sched}"
                        k_o_at_start = f"o LR for ep {epoch}: {self.k_o_lr(epoch-1)}"
                        sigma_sq = f"Sigma squared values at each layer: {self.p.sigma_sq}"
                        alpha = f"Alpha values at each layer: {self.p.alpha}"
                        lam = f"Lambda values at each layer: {self.p.lam}"
                        size_of_starting_img = f"Num params in an original whole input image, regardless of whether images will become tiled for training: {self.p.input_size}"
                        time_created = time_created
                        time_at_chkpt = datetime.datetime.now()
                        train_time_elapsed = time_at_chkpt - time_created
                        time_created_str = f"Time at model creation: {time_created}"
                        time_at_chkpt = f"Time at checkpoint: {time_at_chkpt}"
                        train_time_elapsed = f"Training time elapsed at end of epoch {epoch}: {train_time_elapsed}"

                        metadata_lines = [header, is_tiled, update_scheme, batch_size, epoch_counter, k_r_sched,
                                            k_r_at_start, k_U_sched, k_U_at_start, k_o_sched, k_o_at_start, sigma_sq, alpha, lam, size_of_starting_img,
                                            time_created_str, time_at_chkpt, train_time_elapsed]

                        # Write metadata
                        mod_chkpt_name_txt = mod_chkpt_name + ".txt"
                        with open(mod_chkpt_name_txt, "w") as metadata_out:
                            for line in metadata_lines:
                                metadata_out.write(line)
                                metadata_out.write("\n")

                            metadata_out.write("\n")
                            for line in r_param_lines:
                                metadata_out.write(line)
                                metadata_out.write("\n")

                            for line in U_param_lines:
                                metadata_out.write(line)
                                metadata_out.write("\n")

                            metadata_out.write(o_param_line)
                            metadata_out.write("\n")
                            metadata_out.write(Uo_param_line)
                            metadata_out.write("\n")
                            metadata_out.write(tot_params_line)
                            metadata_out.write("\n")

                            print(f"Trained model metadata at epoch {epoch} {mod_chkpt_name_txt} saved in local dir" + "\n")


            ### Rao and Ballard '99 / Brown, Rogers case: r is allowed to equilibrate before a U matrix receives any information from it
            elif self.update_scheme == "r_eq_then_U":






            print("TRAINING FINISHED" + "\n")
