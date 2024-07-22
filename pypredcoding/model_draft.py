import numpy as np
from learning import constant_lr, step_decay_lr, polynomial_decay_lr
from functools import partial
import math
import data
from parameters import SpccParameters, RpccParameters
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


## Other helpers

def assign_learning_rates(self, component):
    # Dynamically construct the keys
    lr_sched_key = f'lr_{component}_sched'
    k_sched_key = f'k_{component}_sched'
    k_lr_key = f'k_{component}_lr'

    # Access the schedule dictionary dynamically
    lr_sched = list(getattr(self.p, k_sched_key).keys())[0]

    # Assign learning rate functions based on the schedule
    if lr_sched == 'constant':
        setattr(self, k_lr_key, partial(constant_lr, initial=getattr(self.p, k_sched_key)['constant']['initial']))
    elif lr_sched == 'step':
        setattr(self, k_lr_key, partial(step_decay_lr, initial=getattr(self.p, k_sched_key)['step']['initial'], drop_every=getattr(self.p, k_sched_key)['step']['drop_every'], drop_factor=getattr(self.p, k_sched_key)['step']['drop_factor']))
    elif lr_sched == 'poly':
        setattr(self, k_lr_key, partial(polynomial_decay_lr, initial=getattr(self.p, k_sched_key)['poly']['initial'], max_epochs=getattr(self.p, k_sched_key)['poly']['max_epochs'], poly_power=getattr(self.p, k_sched_key)['poly']['poly_power']))


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


class StaticPredictiveCodingClassifier:
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
        # rename it for lower training loop clarity
        n = self.num_nonin_lyrs
        # Total num includes input "layer" (Li case: 4)
        self.num_tot_lyrs = self.num_nonin_lyrs + 1

        # Choices for transformation functions, priors
        self.act_fxn_dict = {'lin':linear_trans,'tan':tanh_trans}
        self.prior_dict = {'gaus':gauss_prior, 'kurt':kurt_prior}

        # Transforms and priors
        self.f = self.act_fxn_dict[self.p.act_fxn]
        self.g = self.prior_dict[self.p.r_prior]
        self.h = self.prior_dict[self.p.U_prior]

        assign_learning_rates(self, 'r')
        assign_learning_rates(self, 'U')
        assign_learning_rates(self, 'o')

        # Avg cost per epoch during training; just representation terms
        self.avg_E_per_ep = []
        # Avg cost per epoch during training; just classification terms
        self.avg_C_per_ep = []
        # Avg prediction error across all layers, avg'd over each image in epoch, during training
        self.avg_PE_all_lyrs_avg_per_ep = []
        # Accuracy per epoch during training
        self.acc_per_ep = []

    def rep_cost(self, label):
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        
        this is called once per image in the training loop
        '''
        
        # squared, sigma-weighted reconstruction error, with priors added below
        E_tot = 0
        E_list = []
        # non squared, non weighted reconstruction error
        PE_list = []

        # We want to track E for Layer 1, Layer 2, Layer 3 (Li 212)
        # this loop will only tackle layer 1 and 2 in a 3 layer model.
        for i in range(1,self.num_nonin_lyrs):
            E_layer = 0
            # Bottom up reconstruction error term, a vector
            bu_err = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
        
            # Top down reconstruction error term, a vector
            td_err = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
            
            # Bottom up error term squared, a scalar
            bu_err_sq = bu_err.T.dot(bu_err)
            # Top down error term squared, also a scalar
            td_err_sq = td_err.T.dot(td_err)
            # Total
            tot_err_sq = bu_err_sq + td_err_sq
            
            # Representation cost E for this layer, is comprised of a bu and td component. (it contains this form for all n-1 layers)
            E_layer = E_layer + ((1 / self.p.sigma_sq[i]) * bu_err_sq) + ((1 / self.p.sigma_sq[i+1]) * td_err_sq)
            E_layer = E_layer + self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0]
            # priors^^^
            '''
            check out sizing of Ui for h later
            '''
            # Store
            E_list.append(E_layer)
            
            # Add layer E to tot E
            E_tot = E_tot + E_layer
            
            # Also calulate bottom up, top down, and total prediction error (ie. L2 norm of the error vector) for each layer
            PE_tot = np.sqrt(tot_err_sq)
            PE_bu = np.sqrt(bu_err_sq)
            PE_td = np.sqrt(td_err_sq)
            # Store
            PE_list.append((PE_tot, PE_bu, PE_td))
            
        # Li 212 Layer 3
        # ie the top layer, the localist layer
        # Bottom up reconstruction error term, a vector
        
        E_layer = 0
        n = self.num_nonin_lyrs
        
        # C1 top layer cost term
        bu_err = self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0]
        td_err = softmax(self.r[n]) - label[:,None] # Difference in order is because not a derivative
        
        # Bottom up error term squared, a scalar
        bu_err_sq = bu_err.T.dot(bu_err)
        # Top down error term squared, also a scalar
        td_err_sq = td_err.T.dot(td_err)
        # Total
        tot_err_sq = bu_err_sq + td_err_sq
        
        # Representation cost E for this layer, is comprised of a bu and td component. (it contains this form for all n-1 layers)
        E_layer = E_layer + ((1 / self.p.sigma_sq[n]) * bu_err_sq) + ((1 / self.p.sigma_sq[n+1]) * td_err_sq)
        E_layer = E_layer + self.h(self.U[n],self.p.lam[n])[0] + self.g(np.squeeze(self.r[n]),self.p.alpha[n])[0]
        # priors^^^
        # Store
        E_list.append(E_layer)
        
        # Add layer E to tot E
        E_tot = E_tot + E_layer
        
        # Also calulate bottom up, top down, and total prediction error (ie. L2 norm of the error vector) for each layer
        PE_tot = np.sqrt(tot_err_sq)
        PE_bu = np.sqrt(bu_err_sq)
        PE_td = np.sqrt(td_err_sq)
        # Store
        PE_list.append((PE_tot, PE_bu, PE_td))
        
        return (E_tot, E_list, PE_list)
    
        # Function to pad a matrix to a target shape
    def pad_to_shape(matrix, target_shape):
        padding = [(0, max(0, t - s)) for s, t in zip(matrix.shape, target_shape)]
        return np.pad(matrix, padding, 'constant')

    
    def rep_cost_stitch(self, label):
        
        def normalized_magnitude(matrix):
            """
            Calculate the normalized magnitude of a matrix.
            
            Parameters:
            - matrix: A numpy array representing the matrix.
            
            Returns:
            - A scalar representing the normalized magnitude of the matrix.
            """
            frobenius_norm = np.linalg.norm(matrix, 'fro')  # Calculate Frobenius norm
            num_elements = matrix.size  # Get the number of elements in the matrix
            normalized_magnitude = frobenius_norm / num_elements  # Normalize by the size of the matrix
            return normalized_magnitude
            
        
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        
        this is called once per image in the training loop
        
        Monica's math/code, adopted to our syntax
        '''
        
        # squared, sigma-weighted reconstruction error, with priors added below
        E_tot = 0
        E_list = []
        # non squared, non weighted reconstruction error
        PE_list = []

        # We want to track E for Layer 1, Layer 2, Layer 3 (Li 212)
        # this loop will only tackle layer 1 and 2 in a 3 layer model.
        for i in range(1,self.num_nonin_lyrs):
            
            
            if i == 1:
            
                E_layer = 0
                # Bottom up reconstruction error term, a vector
                bu_err = (self.r[i-1] - np.matmul(self.U[i], self.r[i][:, :, None]).squeeze())
                # bu_err = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
            
                # Top down reconstruction error term, a vector
                td_err = self.r[i] - self.U[i+1].dot(self.r[i+1]).reshape(self.r[i].shape)
                # td_err = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
                
                # Bottom up error term squared,(ideally) a scalar
                bu_err_sq = normalized_magnitude(bu_err.T.dot(bu_err))
                # Top down error term squared
                td_err_sq = normalized_magnitude(td_err.T.dot(td_err))
                # Total
                tot_err_sq = bu_err_sq + td_err_sq
                
                # if isinstance(tot_err_sq, (int, float)):  # Check if tot_err_sq is a scalar (int or float)
                #     print(f'i = {i}, tot_err_sq is a scalar with value {tot_err_sq}')
                # elif isinstance(tot_err_sq, (list, np.ndarray)):  # Check if tot_err_sq is a list or numpy array (matrix)
                #     print(f'i = {i}, tot_err_sq is a matrix with shape {np.shape(tot_err_sq)} and values:\n{tot_err_sq}')
                # else:
                #     print(f'i = {i}, tot_err_sq is of type {type(tot_err_sq)} with value {tot_err_sq}')
                
                # Representation cost E for this layer, is comprised of a bu and td component. (it contains this form for all n-1 layers)
                E_layer = E_layer + ((1 / self.p.sigma_sq[i]) * bu_err_sq) + ((1 / self.p.sigma_sq[i+1]) * td_err_sq)
            
                E_layer = E_layer + self.h(r_or_U=self.U[i], alph_or_lam=self.p.lam[i])[0] + self.g(r_or_U=self.r[i].squeeze(), alph_or_lam=self.p.alpha[i])[0]
                
                # if isinstance(E_layer, (int, float)):
                #     print(f'i = {i}, E_layer after prior application is a scalar with value {E_layer}')
                # elif isinstance(E_layer, (list, np.ndarray)):
                #     print(f'i = {i}, E_layer after prior application is a matrix with shape {np.shape(E_layer)} and values:\n{E_layer}')
                # else:
                #     print(f'i = {i}, E_layer after prior application is of type {type(E_layer)} with value {E_layer}')
                
                
                
                # priors^^^
                '''
                check out sizing of Ui for h later
                '''
                # Store
                E_list.append(E_layer)
                
                # Add layer E to tot E
                E_tot = E_tot + E_layer
                
                # Also calulate bottom up, top down, and total prediction error (ie. L2 norm of the error vector) for each layer
                PE_tot = np.sqrt(tot_err_sq)
                PE_bu = np.sqrt(bu_err_sq)
                PE_td = np.sqrt(td_err_sq)
                # Store
                PE_list.append((PE_tot, PE_bu, PE_td))
                
                # print(f'td err sq i == {i} is {td_err_sq}')
                
                
            if i >= 2:
            
                E_layer = 0
                # Bottom up reconstruction error term, a vector
                bu_err = self.r[i-1] - self.U[i].dot(self.r[i]).reshape(self.r[i-1].shape)
                # bu_err = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
   
                if len(self.r[i+1].shape) == 1:
                    td_err = self.r[i] - self.U[i+1].dot(self.r[i+1][:, None])
                
                else:
                    # Top down reconstruction error term, a vector
                    td_err = self.r[i] - self.U[i+1].dot(self.r[i+1])
                    
                # td_err = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
                
                # Bottom up error term squared,(ideally) a scalar
                bu_err_sq = normalized_magnitude(bu_err.T.dot(bu_err))
                # Top down error term squared
                td_err_sq = normalized_magnitude(td_err.T.dot(td_err))
                # Total
                tot_err_sq = bu_err_sq + td_err_sq
                
                # if isinstance(tot_err_sq, (int, float)):  # Check if tot_err_sq is a scalar (int or float)
                #     print(f'i = {i}, tot_err_sq is a scalar with value {tot_err_sq}')
                # elif isinstance(tot_err_sq, (list, np.ndarray)):  # Check if tot_err_sq is a list or numpy array (matrix)
                #     print(f'i = {i}, tot_err_sq is a matrix with shape {np.shape(tot_err_sq)} and values:\n{tot_err_sq}')
                # else:
                #     print(f'i = {i}, tot_err_sq is of type {type(tot_err_sq)} with value {tot_err_sq}')
                
                # Representation cost E for this layer, is comprised of a bu and td component. (it contains this form for all n-1 layers)
                E_layer = E_layer + ((1 / self.p.sigma_sq[i]) * bu_err_sq) + ((1 / self.p.sigma_sq[i+1]) * td_err_sq)
                
                # if isinstance(E_layer, (int, float)):
                #     print(f'i = {i}, E_layer after sigma application is a scalar with value {E_layer}')
                # elif isinstance(E_layer, (list, np.ndarray)):
                #     print(f'i = {i}, E_layer after sigma application is a matrix with shape {np.shape(E_layer)} and values:\n{E_layer}')
                # else:
                #     print(f'i = {i}, E_layer after sigma application is of type {type(E_layer)} with value {E_layer}')
                
                
                E_layer = E_layer + self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0]
                
                # if isinstance(E_layer, (int, float)):
                #     print(f'i = {i}, E_layer after prior application is a scalar with value {E_layer}')
                # elif isinstance(E_layer, (list, np.ndarray)):
                #     print(f'i = {i}, E_layer after prior application is a matrix with shape {np.shape(E_layer)} and values:\n{E_layer}')
                # else:
                #     print(f'i = {i}, E_layer after prior application is of type {type(E_layer)} with value {E_layer}')
                
                # print(f'td err sq i == {i} is {td_err_sq}')
                # priors^^^
                '''
                check out sizing of Ui for h later
                '''
                # Store
                E_list.append(E_layer)
                
                # Add layer E to tot E
                E_tot = E_tot + E_layer
                
                # Also calulate bottom up, top down, and total prediction error (ie. L2 norm of the error vector) for each layer
                PE_tot = np.sqrt(tot_err_sq)
                PE_bu = np.sqrt(bu_err_sq)
                PE_td = np.sqrt(td_err_sq)
                # Store
                PE_list.append((PE_tot, PE_bu, PE_td))
                
            
        # Li 212 Layer 3
        # ie the top layer, the localist layer
        # Bottom up reconstruction error term, a vector
        
        E_layer = 0
        n = self.num_nonin_lyrs
        
        # C1 top layer cost term
        
        bu_err = self.r[n-1] - self.U[n].dot(self.r[n]).reshape(self.r[n-1].shape)
        # bu_err = self.r[n-1] - self.U[n].dot(self.r[n])
        
        if len(self.r[n].shape) == 1:
            td_err = softmax(self.r[n]) - label
            
        else:
            td_err = softmax(self.r[n]) - label[:,None] # Difference in order is because not a derivative
        
        # print(f'type of bu_err i == {n} is {type(bu_err)}')
        # print(f'type of td_err i == {n} is {type(td_err)}')
        # print(f'size of td_err i == {n} is {td_err.shape}')
        # print(f'size of bu_err i == {n} is {bu_err.shape}')
        
        # print(f' td err is {td_err}')
        
        
        
        # Bottom up error term squared, a scalar
        bu_err_sq = bu_err.T.dot(bu_err)
        # Top down error term squared, also a scalar
        
        
        td_err_sq = td_err.T.dot(td_err)
        
        if len(td_err_sq.shape) == 2:
            td_err_sq = td_err_sq[0,0]
        elif len(td_err_sq.shape) == 1:
            td_err_sq = td_err_sq[0]
        
        # Total
        tot_err_sq = bu_err_sq + td_err_sq
        
        # print(f'bu err sq i == {n} is {bu_err_sq}')
        # print(f'td err sq i == {n} is {td_err_sq}')
        
        # print(f'size of bu_err_sq i == {n} is {bu_err_sq.shape}')
        # print(f'size of td_err_sq i == {n} is {td_err_sq.shape}')
        
        # sys.exit()
        
        # print(f'type of bu_err_sq i == {n} is {type(bu_err_sq)}')
        # print(f'type of td_err_sq i == {n} is {type(td_err_sq)}')
        # print(f'type of tot_err_sq i == {n} is {type(tot_err_sq)}')
        
        # Representation cost E for this layer, is comprised of a bu and td component. (it contains this form for all n-1 layers)
        E_layer = E_layer + ((1 / self.p.sigma_sq[n]) * bu_err_sq) + ((1 / self.p.sigma_sq[n+1]) * td_err_sq)
        
        # print(f'type of E_layer after sigmas i == {n} is {type(E_layer)}')
        E_layer = E_layer + self.h(self.U[n],self.p.lam[n])[0] + self.g(np.squeeze(self.r[n]),self.p.alpha[n])[0]
        # print(f'type of E_layer after priors i == {n} is {type(E_layer)}')
        
        # priors^^^
        # Store
        E_list.append(E_layer)
        
        # Add layer E to tot E
        E_tot = E_tot + E_layer
        
        # print(f' type of Etot i == {n} is {type(E_tot)}')
        
        # Also calulate bottom up, top down, and total prediction error (ie. L2 norm of the error vector) for each layer
        PE_tot = np.sqrt(tot_err_sq)
        PE_bu = np.sqrt(bu_err_sq)
        PE_td = np.sqrt(td_err_sq)
        # Store
        PE_list.append((PE_tot, PE_bu, PE_td))
        
        return (E_tot, E_list, PE_list)



    def class_cost_nc(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method NC (always 0 with NC: no label data used to train).
        Also, guesses image: returns 0/1 if max arg of softmax(r[n]) doesn't/does match label one-hot elem. """

        n = self.num_nonin_lyrs
        
        # Cost
        NC = 0

        # Guess image
        sm_rn = softmax(self.r[n])

        if np.argmax(sm_rn) == np.argmax(label[:,None]):
            guess_correct_or_not = 1
        else:
            guess_correct_or_not = 0

        return NC, guess_correct_or_not


    def class_cost_c1(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C1. Also, guesses image: returns 0/1 if max arg
        of softmax(r[n]) doesn't/does match label one-hot elem. """

        n = self.num_nonin_lyrs
        
        sm_rn = softmax(self.r[n])

        L = label[:,None]

        # Calc cost (squared euclidian distance (L2 norm)) between the one hot label L
        # and softmax(r_n)
        # This is KB's old way and does not reflect Jc in sPCC math, 2024.07.11
        # C1 = -(1/2)*(label[None,:] - sm_rn).T.dot(label[None,:] - sm_rn)[0,0]

        # Jc in the math is binary crossentropy; Jc = L.T log[softmax(r_n)]
        C1 = L.T.dot(np.log(sm_rn))
        
        if len(C1.shape) == 2:
            C1 = -C1[0,0]
        elif len(C1.shape) == 1:
            C1 = -C1[0]

        # Guess image
        if np.argmax(sm_rn) == np.argmax(L):
            guess_correct_or_not = 1
        else:
            guess_correct_or_not = 0

        return C1, guess_correct_or_not


    def class_cost_c2(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C2. Also, guesses image: returns 0/1 if max arg
        of softmax(r[n]) doesn't/does match label one-hot elem. """
        
        n = self.num_nonin_lyrs

        # Calc cost
        C2 = -1*label[None,:].dot(np.log(softmax((self.U_o.dot(self.r[n])))))[0,0]

        # Guess image
        if np.argmax(softmax(self.r[n])) == np.argmax(label[:,None]):
            guess_correct_or_not = 1
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
        
        n = self.num_nonin_lyrs

        ## Detect WHOLE IMAGE case: model will be constructed with only 1 r[1] module
        if self.p.num_r1_mods == 1:

            '''
            not yet built
            '''

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
            
            # NOTE: should take X as num imgs * num tiles per image, tlx pixels * tly pixels

            # Set some general attrs
            self.is_tiled = True
            # X dims for Li case should be: (1125, 256); thus sgl_tile_area 256
            # This is 225 tiles (15x15 tiles) per image; each having an area of 16x16y pixels
            
            # X dim for Li 212 case should be 3392, 864
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
        
        # For naming only, inits only
        # Set learning rates at the start of each epoch
        k_r = self.k_r_lr(0)
        k_U = self.k_U_lr(0)
        k_o = self.k_o_lr(0)
        
        model_pkl_name = model_name_untrained + f"_kr{k_r}kU{k_U}" + ".pydb"
        model_metadata_name = model_name_untrained + f"_kr{k_r}kU{k_U}" + ".txt"

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

        """
        WHOLE IMAGE TRAINING CASE
        """

        #### WHOLE IMAGE TRAINING case
        if self.is_tiled is False:
            print("Model training on NON-TILED input")
            print("non-tiled not yet written")

            if self.p.update_scheme == "rU_simultaneous":
                print("non-tiled rU simultaneous not yet written, quitting...")
                exit()

            elif self.p.update_scheme == "r_eq_then_U":
                print("non-tiled r eq then U not yet written, quitting...")
                exit()

            exit()

        """
        TILED IMAGE TRAINING CASE
        """

        #### TILED TRAINING case
        if self.is_tiled is True:
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


            ### TRAINING DATA COLLECTION INITS

            # Initiate training_wide E collection list (will contain E contributions of each image for epoch)
            self.E_contrib_per_img_all_eps = []
            # Each E_tot should == sum of each E_contrib across all imgs in a single epoch
            self.E_tot_per_ep_all_eps = []
            # Training wide E collection list (Es by layer, image and epoch)
            self.Es_by_lyr_per_img_all_eps = []
            # Initiate training-wide PE collection list (will contain PEs by layer, image and epoch)
            self.PEs_by_lyr_per_img_all_eps = []
            # Initiate training_wide C collection list (will contain C contributions of each image for epoch)
            self.C_contrib_per_img_all_eps = []
            # Each C_tot should == sum of each C_contrib across all imgs in a single epoch
            self.C_tot_per_ep_all_eps = []
            # Initiate training-wide correct classifications (0 = miss; 1 = hit) tracker (across all images and epochs)
            self.classif_success_per_img_all_eps = []
            # Initiate training-wide classification Accuracy collection list (across all epochs)
            self.classif_accuracy_all_eps = []
            # Initiate tracking list of all randomized index sets for training set shuffling (1 idx set for each epoch)
            # Epoch 'zero' is not randomized.
            self.randomized_training_indices_all_eps = []
            
            print('size of r[1] is (before rep cost call ep 0)', self.r[1].shape)
            print('size of r[2] is (before rep cost call ep 0)', self.r[2].shape)
            
            ### Li case: and updating proceeds through layers, r and U of a layer i are updated simultaneously 30 times
            ### This means each each layer's r/U updates 30 times per image, using top down and bottom up information specific to that image
            if self.p.update_scheme == "rU_simultaneous":
                print("rU simultaneous TRAINING about to begin")

                self.num_rUsimul_iters = 100

                #### EPOCH "0" CALCULATIONS (E, PE WITH ALL INITIAL, RANDOMIZED ARRAYS)
                # Functions as negative (pre-update) control
                
                # Initiate epoch-dependent measures
                # Total error
                E_tot_sgl_ep = 0
                # Error by image
                E_contrib_per_img_sgl_ep = []
                # Es by layer
                Es_by_lyr_per_img_sgl_ep = []
                # Classification error only
                C_tot_sgl_ep = 0
                # Classification error by image
                C_contrib_per_img_sgl_ep = []
                # Prediction error (part of E, highly correlated)
                PEs_by_lyr_per_img_sgl_ep = []
                # Number of correct classifications
                classif_success_per_img_sgl_ep = []

                for image in range(0,self.num_training_imgs):
                    
                    single_image_tileset = X[image]
                    self.r[0] = single_image_tileset

                    # rep_cost returns a tuple of E, E_list (E's by layer), PE_list (PEs by layer)
                    Eimg_Elistimg_and_PEsimg = self.rep_cost_stitch(Y[image])
                    # Loss (E) for random image "0"
                    Eimg = Eimg_Elistimg_and_PEsimg[0]
                    # Loss by layers
                    Elist = Eimg_Elistimg_and_PEsimg[1]
                    # E total for epoch 0
                    E_tot_sgl_ep = E_tot_sgl_ep + Eimg
                    # Add E contrib
                    E_contrib_per_img_sgl_ep.append(Eimg)
                    # Add e by layer
                    Es_by_lyr_per_img_sgl_ep.append(Elist)
                    
                    # PEs for each layer for random image "0"
                    PEs_by_lyr_sgl_img = Eimg_Elistimg_and_PEsimg[2]
                    
                    # print(f' type of Eimg is {type(Eimg)}')
                    # print(f' type of Elist is {type(Elist)}')
                    # print(f' type of PEs_by_lyr_sgl_img is {type(PEs_by_lyr_sgl_img)}')
                    
                    
                    # Add PEs to beginning of tracker list
                    PEs_by_lyr_per_img_sgl_ep.append(PEs_by_lyr_sgl_img)
                    
                    # classification costs
                    Cimg, guess_correct_or_not = class_cost(Y[image])
                    # Add to total
                    C_tot_sgl_ep = C_tot_sgl_ep + Cimg
                    # Add C contrib
                    C_contrib_per_img_sgl_ep.append(Cimg)
                    classif_success_per_img_sgl_ep.append(guess_correct_or_not)
                    
                    # Reset r[1], r[2], r[3(n)] to zero for the next image
                    self.r[1] = np.zeros((self.p.num_r1_mods, self.p.hidden_sizes[0]))
                    self.r[2] = np.zeros(self.p.hidden_sizes[1])
                    # "Localist" layer (relates size of Y (num classes) to final hidden layer)
                    self.r[3] = np.zeros((self.p.output_size, 1))
                    
                    

                # Add epoch 0 E to beginning of tracker list
                self.E_contrib_per_img_all_eps.append(E_contrib_per_img_sgl_ep)
                self.E_tot_per_ep_all_eps.append(E_tot_sgl_ep)
                self.Es_by_lyr_per_img_all_eps.append(Es_by_lyr_per_img_sgl_ep)
                
                # Add epoch 0 PEs to beginning of tracker list
                self.PEs_by_lyr_per_img_all_eps.append(PEs_by_lyr_per_img_sgl_ep)
                
                # add epoch 0 C to beginning of tracker list
                self.C_contrib_per_img_all_eps.append(C_contrib_per_img_sgl_ep)
                self.C_tot_per_ep_all_eps.append(C_tot_sgl_ep)
                
                # add epoch 0 classif success to beginning of tracker list
                self.classif_success_per_img_all_eps.append(classif_success_per_img_sgl_ep)
                
                # acc
                accuracy_sgl_ep = np.sum(classif_success_per_img_sgl_ep) / self.num_training_imgs
                self.classif_accuracy_all_eps.append(accuracy_sgl_ep)
                # Epoch 'zero' is not randomized.
                self.randomized_training_indices_all_eps.append(np.arange(0,self.num_training_imgs))
                
                print(f"Epoch 0 (pre training, unshuffled images) accuracy is {accuracy_sgl_ep}" + "\n")
                print(f"Epoch 0  E is {E_tot_sgl_ep}" + "\n")
                print(f"Epoch 0  C is {C_tot_sgl_ep}" + "\n")
                
                print(f"X.shape is {X.shape}")
                single_image_tileset = X[0]
                print(f"single_image_tileset.shape is {single_image_tileset.shape}")
                label = Y[0]
                print(f"single label.shape image {1} is {label.shape}")
                
                #### EPOCHS 1 - n

                for epoch in range(1, self.p.num_epochs + 1):

                    print(f"Epoch: {epoch}")

                    # Shuffle indices of X, Y together, each epoch
                    N_permuted_indices = np.random.permutation(self.num_training_imgs)
                    X_shuffled = X[N_permuted_indices]
                    Y_shuffled = Y[N_permuted_indices]

                    ### Initialize epoch-dependent measures
                    
                    # Total error
                    E_tot_sgl_ep = 0
                    # Error by image
                    E_contrib_per_img_sgl_ep = []
                    # error by layer per image
                    Es_by_lyr_per_img_sgl_ep = []
                    # Classification error only
                    C_tot_sgl_ep = 0
                    # Classification error by image
                    C_contrib_per_img_sgl_ep = []
                    # Prediction error (part of E, highly correlated)
                    PEs_by_lyr_per_img_sgl_ep = []
                    # Number of correct classifications
                    classif_success_per_img_sgl_ep = []
                    

                    # Set learning rates at the start of each epoch
                    k_r = self.k_r_lr(epoch-1)
                    k_U = self.k_U_lr(epoch-1)
                    k_o = self.k_o_lr(epoch-1)

                    """
                    # internal tile plotting code existed here pre 2024.07.10
                    """

                    #### GRADIENT DESCENT LOOP
                    # NOTE: (only batch size 1 currently supported)

                    for image in range(0, self.num_training_imgs):
                        
                        # print(f'epoch {epoch} image {image}')

                        ## Parse out tileset and label for one image
                        # single image tilset is a set of n=self.num_tiles_per_img tiles; Li case: shape (225, 256)
                        single_image_tileset = X_shuffled[image]
                        # This single image's tileset becomes the model's "0th"-level representation
                        # I.e. is the original image, in tiled form, with user-tailored rf settings
                        self.r[0] = single_image_tileset
                        
                        # Set matching label
                        label = Y_shuffled[image]

                        '''
                        internal plotting test for shuffling 
                        of tiled images was here,
                        pre 2024.07.10
                        '''
                        
                        '''
                        original
                        unstitched
                        '''

                        # for iteration in range(0,self.num_rUsimul_iters):
                            
                        #     ### r loop (splitting r loop, U loop mimic's Li architecture)
                        #     ### (i ... n-1)
                        #     for i in range(1, n):

                        #         # r update
                        #         self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                        #         * self.U[i].T.dot(self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1]) \
                        #         + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                        #         - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

                        #     # final r (Li's "localist") layer update
                        #     self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                        #     * self.U[n].T.dot(self.f(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])[1]) \
                        #     - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

                        #     # later: change based on C1, C2 or NC setting
                        #     # C1 for now
                        #     # size eg 212,1 label , 212,1 r[n]
                        #     # only one r learning rate in Li 212.
                        #     + ((k_r) * (label[:,None] - softmax(self.r[n])))

                        #     ### U loop ( i ... n)
                        #     for i in range(1, n+1):

                        #         # U update
                        #         self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                        #         * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
                        #         - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
                        
                        '''
                        end orig, unstitched
                        '''
                        
                        
                        '''
                        stitch Monica's code here into our syntax.
                        '''
                        
                        for iteration in range(0,self.num_rUsimul_iters):
                            
                            ### r loop (splitting r loop, U loop mimic's Li architecture)
                            ### (i ... n-1)
                            
                            
                            for i in range(1, n):
                            
                                if i == 1:

                                    # r update
                                    self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                                    * np.matmul(np.transpose(self.U[i], axes=(0,2,1)), (self.r[i-1] - np.matmul(self.U[i], self.r[i][:, :, None]).squeeze())[:, :, None]).squeeze()
                                    + (k_r/self.p.sigma_sq[i+1]) * -(self.r[i] - self.U[i+1].dot(self.r[i+1]).reshape(self.r[i].shape))
                                    - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]
                                    
                                elif i == 2:
                                    
                                    # r update
                                    self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                                    * self.U[i].T.dot((self.r[i-1] - (self.U[i].dot(self.r[i])).reshape((16,32))).reshape(512)) \
                                    + (k_r / self.p.sigma_sq[i+1]) * -(self.r[i] - (self.U[i+1].dot(self.r[i+1])).squeeze()) \
                                    - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]
                                    
                                else:

                                    # r update
                                    self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                                    * self.U[i].T.dot(self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1]) \
                                    + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                                    - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]
                            
                            # final r (Li's "localist") layer update
                            self.r[n] = self.r[n].squeeze()
                            self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                            * self.U[n].T.dot(self.r[n-1] - self.U[n].dot(self.r[n])) \
                            - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

                            # later: change based on C1, C2 or NC setting
                            # C1 for now
                            # size eg 212, label , 212, r[n]
                            # only one r learning rate in Li 212.
                            + ((k_r) * (label - softmax(self.r[n])))
                            
                            
                            # td_err_online = (k_r) * (label - softmax(self.r[n]))
                            # td_err_sq = td_err_online.T.dot(td_err_online)
                            # if len(td_err_sq.shape) == 2:
                            #     td_err_sq = td_err_sq[0,0]
                            # elif len(td_err_sq.shape) == 1:
                            #     td_err_sq = td_err_sq[0]
                            # print(f'online  "C" in rn update (td component L2 norm) is {td_err_sq}')

                            ### U loop ( i ... n)
                            for i in range(1, n+1):
                                
                                if i == 1:

                                    # U update
                                    self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                                    * np.matmul((self.r[i-1] - np.matmul(self.U[i], self.r[i][:, :, None]).squeeze())[:, :, None], self.r[i][:, None, :]) \
                                    # * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
                                    - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
                                    
                                elif i == 2:
                                    
                                    # U update
                                    self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                                    * np.tile((np.matmul((self.r[i-1] - np.matmul(self.U[i], self.r[i][:, None]).squeeze().reshape((16,32))), self.r[i].reshape((32,4)))).flatten(), 2)\
                                    # * np.outer((self.r[i-1] - (self.U[i].dot(self.r[i])).reshape(self.r[i-1].shape)).flatten(), self.r[i]) \
                                    # * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
                                    - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
                                    
                                else:
                                    
                                    # U update
                                    self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                                    * np.outer(self.r[i-1] - self.U[i].dot(self.r[i]), self.r[i].T) \
                                    - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
                                    
                        '''
                        end stich
                        '''
                        

                        ### Training loss function E and PE by layer
                        # rep_cost returns a tuple of E, PE_list (PEs by layer for that image)
                        Eimg_Elist_and_PEsimg = self.rep_cost_stitch(label)

                        # Loss (E) for this image
                        Eimg = Eimg_Elist_and_PEsimg[0]
                        Elist = Eimg_Elist_and_PEsimg[1]
                        # Track E contribution by image across whole epoch
                        E_contrib_per_img_sgl_ep.append(Eimg)
                        Es_by_lyr_per_img_sgl_ep.append(Elist)
                        # Add single image's representation cost to epoch's E total
                        E_tot_sgl_ep = E_tot_sgl_ep + Eimg

                        # PEs for each layer for this image
                        PEs_by_lyr_sgl_img = Eimg_Elist_and_PEsimg[2]
                        # Track PEs by layer across whole epoch
                        PEs_by_lyr_per_img_sgl_ep.append(PEs_by_lyr_sgl_img)

                        # Classification function C (calls relevant fxn: C1, C2, or zero multiplier for NC)
                        # Calculates Cimg and tracks label guess success (0) or failure (1)
                        Cimg, guess_correct_or_not = class_cost(label)
                        # Track C contribution by image across whole epoch
                        C_contrib_per_img_sgl_ep.append(Cimg)
                        # Add single image's classif cost to epoch's C total (should go down over time)
                        C_tot_sgl_ep = C_tot_sgl_ep + Cimg
                        # Count correct/incorrect classification guesses over epoch for accuracy per epoch
                        classif_success_per_img_sgl_ep.append(guess_correct_or_not)
                        
                        # Maybe later: track EC (E + C which is the actual loss being minimized)

                        # Reset r[1], r[2], r[3(n)] to zero for the next image
                        self.r[1] = np.zeros((self.p.num_r1_mods, self.p.hidden_sizes[0]))
                        self.r[2] = np.zeros(self.p.hidden_sizes[1])
                        # "Localist" layer (relates size of Y (num classes) to final hidden layer)
                        self.r[3] = np.zeros((self.p.output_size, 1))
                        

                    ### STORE SALIENT TRAINING DATA AS ATTRIBUTES
                    # Store this epoch's E contributions per image
                    self.E_contrib_per_img_all_eps.append(E_contrib_per_img_sgl_ep)
                    # Store this epoch's total E after each Eimg
                    self.E_tot_per_ep_all_eps.append(E_tot_sgl_ep)
                    # epochs layer based E
                    self.Es_by_lyr_per_img_all_eps.append(Es_by_lyr_per_img_sgl_ep)
                    # Store this epoch's PEs (parsed per layer, per image)
                    self.PEs_by_lyr_per_img_all_eps.append(PEs_by_lyr_per_img_sgl_ep)
                    # Store this epoch's C contributions per image
                    self.C_contrib_per_img_all_eps.append(C_contrib_per_img_sgl_ep)
                    # Store this epoch's total C after each Cimg
                    self.C_tot_per_ep_all_eps.append(C_tot_sgl_ep)
                    # Store this epoch's correct / incorrect guess markers (i.e. 1 or 0) by image
                    self.classif_success_per_img_all_eps.append(classif_success_per_img_sgl_ep)
                    # Calculate classification accuracy for this epoch
                    accuracy_sgl_ep = np.sum(self.classif_success_per_img_all_eps[epoch-1]) / self.num_training_imgs
                    # Store this epoch's accuracy measurement
                    self.classif_accuracy_all_eps.append(accuracy_sgl_ep)
                    # Store this epoch's randomized indices to be able to backtrack desired image-specific data from each above list
                    self.randomized_training_indices_all_eps.append(N_permuted_indices)


                    print(f"Epoch {epoch} (while training, shuffled images) accuracy is {accuracy_sgl_ep}" + "\n")
                    print(f"Epoch {epoch}  E is {E_tot_sgl_ep}" + "\n")
                    print(f"Epoch {epoch}  C is {C_tot_sgl_ep}" + "\n")


                    # Checkpointing logic
                    if epoch % chkpt_every_n == 0:

                        # Pickle model
                        mod_chkpt_name = model_name_pre_epoch + str(epoch)
                        mod_chkpt_name_pkl = mod_chkpt_name + f"_kr{k_r}kU{k_U}" + ".pydb"

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
                        
                        # NOTE: is this true?
                        # p.input_size = tile area, I thought.
                        size_of_starting_img = f"UNTESTED, could just be tile: Num params in an original whole input image, regardless of whether images will become tiled for training: {self.p.input_size}"
                        
                        time_created = time_created
                        time_at_chkpt = datetime.datetime.now()
                        train_time_elapsed = time_at_chkpt - time_created
                        time_created_str = f"Time at model creation: {time_created}"
                        time_at_chkpt = f"Time at checkpoint: {time_at_chkpt}"
                        train_time_elapsed = f"Training time elapsed at end of epoch {epoch}: {train_time_elapsed}"
                        accuracy_at_chkpt = f"Accuracy at end of epoch {epoch}: {accuracy_sgl_ep}"

                        metadata_lines = [header, is_tiled, update_scheme, batch_size, epoch_counter, k_r_sched,
                                            k_r_at_start, k_U_sched, k_U_at_start, k_o_sched, k_o_at_start, sigma_sq, alpha, lam, size_of_starting_img,
                                            time_created_str, time_at_chkpt, train_time_elapsed, accuracy_at_chkpt]

                        # Write metadata
                        mod_chkpt_name_txt = mod_chkpt_name + f"_kr{k_r}kU{k_U}" + ".txt"
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

                print("rU_simultaneous epochs-loop finished")

            ### Rao and Ballard '99 / Brown, Rogers case: r is allowed to equilibrate before a U matrix receives any information from it
            elif self.p.update_scheme == "r_eq_then_U":

                print("r_eq_then_U epochs-loop finished")

            print("TRAINING FINISHED" + "\n")
