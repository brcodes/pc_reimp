# Implementation of r&b 1999 predictive coding model with MNIST data.

import numpy as np
from matplotlib import pyplot as plt
# from kbutil.plotting import pylab_pretty_plot
from learning import *
from functools import partial
import math
import data
import cv2

# activation functions
def linear_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Linear transformation. """
    f = U_dot_r
    F = np.eye(len(f))
    return (f, F)



def tanh_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Tanh transformation. """
    f = np.tanh(U_dot_r)
    F = np.diag(1 - f.flatten()**2)
    return (f, F)



# r, U prior functions
def gauss_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior. """
    g_or_h = alph_or_lam * np.square(r_or_U).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)



def kurt_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior. """
    g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    return (g_or_h, gprime_or_hprime)



# softmax function
def softmax(r):
    return np.exp(r) / np.exp(r).sum()



class PredictiveCodingClassifier:
    def __init__(self, parameters):

        self.p = parameters

        # possible choices for transformations, priors
        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # # NOTE: may use this functionality later
        # # possible classification methods
        # self.class_cost_dict = {'C1':class_c1, 'C2':class_c2}

        # all the representations (including the image r[0] which is not trained)
        self.r = {}

        # synaptic weights controlling reconstruction in the network
        self.U = {}

        # priors and transforms
        self.f = self.unit_act[self.p.unit_act]
        # how to call f(x): self.f(self.U.dot(self.r))[0]
        # how to call F(x): self.f(self.U.dot(self.r))[1]

        self.g = self.prior_dict[self.p.r_prior]
        # how to call g(r): self.g(self.r,self.p.alpha)[0]
        # how to call g'(r): self.g(self.r,self.p.alpha)[1]

        self.h = self.prior_dict[self.p.U_prior]
        # how to call h(U): self.h(self.U,self.p.lam)[0]
        # how to call h'(U): self.h(self.U,self.p.lam)[1]


        # # classification method
        # self.class_cost = self.class_cost_dict[self.p.classification]
        # # if C1, how to call C1: C = C - self.class_cost(self.r[n], label)
        # # if C2, how to call C2: C = self.class_cost(self.r[n], self.U_o, label) + \
        # # (self.h(self.U_o,self.p.lam[n-1])[0])[0,0]

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


        #make initial learning rate self.variables for plotting.py
        self.lr_r = list(list(self.p.k_r_sched.values())[0].values())[0]
        self.lr_U = list(list(self.p.k_U_sched.values())[0].values())[0]
        self.lr_o = list(list(self.p.k_o_sched.values())[0].values())[0]

        # total number of layers (Input, r1, r2... rn, Output)
        self.n_layers = len(self.p.hidden_sizes) + 2
        # number of non-image layers (r1, r2... rn, Output)
        self.n_non_input_layers = len(self.p.hidden_sizes) + 1
        # number of hidden layers (r1, r2... rn)
        self.n_hidden_layers = len(self.p.hidden_sizes)

        # initialize appropriate r's and U's
        # N.B - MAY NEED TO SET THE FOLLOWING PROPERLY USING PRIORS AT SOME POINT

        # initialize image input layer of size (input_size,)
        self.r[0] = np.random.randn(self.p.input_size,1)

        print('\n')
        print("*** Predictive Coding Classifier ***")
        print('\n')
        print("*** Layer shapes ***")
        print('\n')
        print("r[0] shape is " + str(np.shape(self.r[0])))
        print('\n')

        # dict to contain number of neural network parameters per layer comprising the model
        self.nn_parameters_dict = {}

        # initialize r's and U's for hidden layers
        # calculate number of network parameters per layer
        for i in range(1,self.n_non_input_layers):
            self.r[i] = np.random.randn(self.p.hidden_sizes[i-1],1)
            self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))
            ri = "r_{}".format(i)
            Ui = "U_{}".format(i)
            self.nn_parameters_dict[ri] = len(self.r[i])
            self.nn_parameters_dict[Ui] = np.shape(self.U[i])[0]*np.shape(self.U[i])[1]

            print("r[{}] shape is ".format(i) + str(np.shape(self.r[i])))
            print('\n')
            print("U[{}] shape is ".format(i) + str(np.shape(self.U[i])))
            print('\n')

        print("len(self.r): " + str(len(self.r)))
        print('\n')

        # initialize "output" layer
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        print("o shape is " + str(np.shape(self.o)))
        print('\n')
        print("U_o shape is " + str(np.shape(self.U_o)))
        print('\n')

        # calculate total number of network parameters comprising the model (hidden layers only)
        self.n_model_parameters = sum(self.nn_parameters_dict.values())

        print("*** Network Parameters ***")
        print('\n')
        print("number of neural network parameters per hidden layer: " + str(self.nn_parameters_dict))
        print('\n')
        print("total number of hidden layer parameters: {}".format(self.n_model_parameters))
        print('\n')

        # initialize number of training images, prediction images, and prediction updates
        # will stay 0 if model untrained, or if predict() has not yet been used on it, respectively
        self.n_training_images = 0
        self.n_pred_images = 0
        self.n_pred_updates = 0

        # all below will stay empty lists if model has not been trained,
        # or if predict() has not yet been used on it, respectively

        # average cost per epoch during training; just representation terms
        self.E_avg_per_epoch = []
        # average cost per epoch during training; just classification term
        self.C_avg_per_epoch = []
        # accuracy per epoch during training
        self.acc_per_epoch = []
        # actual prediction vectors (r[n]'s)
        self.prediction = []

        return



    def rep_cost(self):
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        '''
        E = 0
        # LSQ cost
        for i in range(0,len(self.r)-1):
            v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
            E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
        # priors on r[1],...,r[n]; U[1],...,U[n]
        for i in range(1,len(self.r)):
            E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])
        return E



    def class_cost_1(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C1. """
        n = self.n_hidden_layers
        C1 = -(1/2)*(label[None,:] - softmax(self.r[n])).T.dot(label[None,:] - softmax(self.r[n]))[0,0]
        return C1



    def class_cost_2(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C2, uninclusive of the prior term. """
        n = self.n_hidden_layers
        C2 = -1*label[None,:].dot(np.log(softmax((self.U_o.dot(self.r[n])))))[0,0]
        return C2

    def prediction_error(self,layer_number):
        """ Calculates the normed prediction error of a layer (in [i,n]), i.e. the difference between a
        the layer's image prediction ("r^td") and the image representation from the layer below ("r"). """
        pe = math.sqrt((self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]).T\
        .dot(self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]))
        return pe



    def train(self,X,Y):
        '''
        X: matrix of input patterns (N_patterns x input_size)
        Y: matrix of output/target patterns (N_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        # number of hidden layers
        n = self.n_hidden_layers

        # initialize "output" layer o (for classification method 2 (C2))
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights U_o to the output (C2)
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        print("*** Training ***")
        print('\n')

        # loop through training image dataset num_epochs times
        for epoch in range(0,self.p.num_epochs):
            # shuffle order of training set input image / output vector pairs each epoch
            N_permuted_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[N_permuted_indices]
            Y_shuffled = Y[N_permuted_indices]

            # print("y_shuffled shape is: " + '\n' + str(Y_shuffled.shape))

            # number of training images
            self.n_training_images = X_shuffled.shape[0]

            # we compute average cost per epoch (batch size = 1); separate classification
            #   and representation costs so we can compare OOM sizes
            E = 0
            C = 0

            # accuracy per epoch: how many images are correctly guessed per epoch
            self.num_correct = 0

            # set learning rates at the start of each epoch
            k_r = self.k_r_lr(epoch)
            k_U = self.k_U_lr(epoch)
            k_o = self.k_o_lr(epoch)

            # print("*** train() function values and layers ***")
            # print("Number of training images is {}".format(num_images) + '\n')

            print("epoch {}".format(epoch+1))

            # loop through training images
            for image in range(0, self.n_training_images):


                # copy first image into r[0]
                single_image = X_shuffled[image,:][:,None]
                self.r[0] = single_image
                # print('shape of self.r[0] is {}'.format(self.r[0].shape))
                #reshape for predict function
                reshaped_image = single_image.reshape(28,28)
                # print('reshaped image size is {}'.format(reshaped_image.shape))

                #Initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    # self state per layer
                    self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                    
                #Update them for 100 iterations using self.predict() to ensure U's are trained with
                #optimal r's
                
                self.predict(reshaped_image)
                print('prediction C1 {} epoch {} finished'.format(image+1, epoch+1))
                
                # designate label vector
                label = Y_shuffled[image,:]


                # loop through intermediate layers (will fail if number of hidden layers is 1)
                # r,U updates written symmetrically for all layers including output
                for i in range(1,n):


                    # NOTE: self.p.k_r learning rate
                    # r[i] update
                    self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                    * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                    + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                    - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]


                    # U[i] update
                    self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                    * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                    - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]


                self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

                # classification term C1
                + ((k_o) * (label[:,None] - softmax(self.r[n])))

                # U[n] update (C1, C2) (identical to U[i], except index numbers)
                self.U[n] = self.U[n] + (k_U / self.p.sigma_sq[n]) \
                * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                - (k_U / 2) * self.h(self.U[n],self.p.lam[n])[1]
                
                print('update C1 {} epoch {} finished'.format(image+1, epoch+1))


                # Loss function E
                E = E + self.rep_cost()

                # Classification cost function C


                """ Classifying using C1 """
                C = (self.class_cost_1(label))
                E = E + C
                self.class_type = 'C1'

                # Classification Accuracy

                # if index of the max value in final representation vector
                # = index of the 1 in associated one hot label vector
                # then the model classified the image correctly
                # and add 1 to num_correct


                """ C1 method """
                if np.argmax(softmax(self.r[n])) == np.argmax(label[:,None]):
                    self.num_correct += 1

                # print("Image {} epoch {}".format(image+1,epoch+1))
                # print('\n')
                # # print("self.r[1][:5]")
                # print(self.r[1][:5])
                # print('\n')
                # print("self.r[n][:2]")
                # print(self.r[n][:2])
                # print('\n')
                # print("self.U[1][:2]")
                # print(self.U[1][:2])
                # print('\n')
                # print("self.U[2][:2]")
                # print(self.U[2][:2])
                # print('\n')


            print('self.num_correct non tiled epoch {}'.format(epoch+1))
            print(self.num_correct)

            # store average costs and accuracy per epoch
            E_avg_per_epoch = E/self.n_training_images
            C_avg_per_epoch = C/self.n_training_images
            acc_per_epoch = round((self.num_correct/self.n_training_images)*100)

            self.E_avg_per_epoch.append(E_avg_per_epoch)
            self.C_avg_per_epoch.append(C_avg_per_epoch)
            self.acc_per_epoch.append(acc_per_epoch)

        return


    def predict(self,X):
        '''
        Given one or more inputs, produce one or more outputs. X should be a matrix of shape [n_pred_images,:,:]
        or a single image of size [:,:]. Predict returns a list of predictions (self.prediction), i.e.
        self.prediction = [[contents_of_r[n]_img1],[contents_of_r[n]_img2]]. Therefore,
        self.prediction[0] = the actual vector of interest (what the model "sees") = [contents_of_r[n]_img1]
        predict() also saves a list of per-update-PEs for each image, split by layer.
        If you predict (e.g.) 2 images, accessing these PEs is as follows:
        image1_layer1PE, image1_layer2PE = self.prediction_errors_l1[0], self.prediction_errors_l2[0]
        image2_layer1PE, image2_layer2PE = self.prediction_errors_l1[1], self.prediction_errors_l2[1]
        '''

        # number of hidden layers
        n = self.n_hidden_layers

        # number of r updates before r's "relax" into a stable representation of the image
        # empirically, between 50-100 seem to work, so we'll stick with 100.

        self.n_pred_updates = 100
        # re-initialize lists of actual prediction outputs and PEs
        self.r1s = []
        self.prediction = []
        self.prediction_errors_l1 = []
        self.prediction_errors_l2 = []

        # set learning rate for r
        k_r = 0.05


        # if X is a matrix of shape [n_pred_images,:,:].
        # i.e if the input is multiple images
        if len(X.shape) == 3:

            print("using predict(3-dim_vec_input)")

            self.n_pred_images = X.shape[0]
            print("npredimages")
            print(self.n_pred_images)

            # get from [n,28,28] input to [n,784] so that self.r[0] instantiation below
            # can convert to and call each image as a [784,1] vector
            X_flat = data.flatten_images(X)


            print("*** Predicting ***")
            print('\n')

            # loop through testing images
            for image in range(0, self.n_pred_images):

                print("starting image {}".format(image+1))

                # representation costs for zeroth and nth layers
                self.pe_1 = []
                self.pe_2 = []

                # copy first image into r[0]

                # print(X[image].shape)
                # convert [1,784] image to one [784,1] image
                self.r[0] = X_flat[image,:][:,None]

                # print(X[image].shape)
                # print(X[image][:,None].shape)
                # print(self.r[0].shape)
                # print("fUr shape")
                # print((self.f(self.U[1].dot(self.r[1]))[0]).shape)


                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    # self state per layer
                    self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                    # print('rlayer')
                    # print(self.r[layer].shape)


                for update in range(0,self.n_pred_updates):

                    # magnitude (normed) prediction errors each "layer" (i.e. error between r0,r1, and r1,r2)

                    pe_1 = self.prediction_error(1)
                    pe_2 = self.prediction_error(2)

                    self.pe_1.append(pe_1)
                    self.pe_2.append(pe_2)

                    # loop through intermediate layers (will fail if number of hidden layers is 1)
                    # r,U updates written symmetrically for all layers including output
                    for i in range(1,n):


                        # r[i] update
                        self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                        * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                        + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                        - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]


                    self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                    * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                    - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1]


                # return final predictions
                # i.e. r[n]'s

                r1 = self.r[1]
                prediction = self.r[n]

                self.r1s.append(r1)
                self.prediction.append(prediction)
                self.prediction_errors_l1.append(self.pe_1)
                self.prediction_errors_l2.append(self.pe_2)


        # if X is a single image
        # of shape [:,:]
        elif len(X.shape) == 2:

            # print("Xshape is")
            # print(X.shape)

            self.n_pred_images = 1

            # get from [28,28] input to [1,784] so that self.r[0] instantiation below
            # can convert to and call the image as a [784,1] vector
            X_flat = data.flatten_images(X[None,:,:])

            # print("Xflat is")
            # print(X_flat.shape)

            print('\n')
            print("*** Predicting single image ***")

            # loop through testing images
            for image in range(0, self.n_pred_images):

                # representation costs for zeroth and nth layers
                self.pe_1 = []
                self.pe_2 = []

                # copy first image into r[0]

                # print(X[image].shape)
                # convert [1,784] image to one [784,1] image
                self.r[0] = X_flat[image,:][:,None]

                # print(X[image].shape)
                # print(X[image][:,None].shape)
                # print(self.r[0].shape)
                # print("fUr shape")
                # print((self.f(self.U[1].dot(self.r[1]))[0]).shape)


                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    # self state per layer
                    self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                    # print('rlayer')
                    # print(self.r[layer].shape)


                for update in range(0,self.n_pred_updates):

                    # magnitude (normed) prediction errors each "layer" (i.e. error between r0,r1, and r1,r2)

                    pe_1 = self.prediction_error(1)
                    pe_2 = self.prediction_error(2)

                    self.pe_1.append(pe_1)
                    self.pe_2.append(pe_2)

                    # loop through intermediate layers (will fail if number of hidden layers is 1)
                    # r,U updates written symmetrically for all layers including output
                    for i in range(1,n):


                        # r[i] update
                        self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                        * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                        + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                        - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]


                    self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                    * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                    - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1]


                # return final prediction (r[n]) and final r[1]

                r1 = self.r[1]
                prediction = self.r[n]

                self.r1s.append(r1)
                self.prediction.append(prediction)
                self.prediction_errors_l1.append(self.pe_1)
                self.prediction_errors_l2.append(self.pe_2)

        else:
            print("input vector must be 2 or 3-dim")

        return self.prediction

    def evaluate(self,X,Y,eval_class_type='C1'):

        """ evaluates model's E, C and classification accuracy in any state (trained, untrained)
        using any input data. X should be a matrix of shape [n_pred_images,:,:] or a single image of size [:,:]
        Calls self.predict(): predict can take a 3-dim (multi-image) or 2-dim (single image) vector, but when called
        here in evalute(), predict only takes in one image (2-dim vec) at a time."""

        self.E_per_image = []
        self.C_per_image  = []
        self.Classif_success_by_img = []
        self.acc_evaluation = 0
        self.eval_class_type = eval_class_type
        self.softmax_guess_each_img = []

        # if X is a matrix of shape [n_eval_images,:,:].
        # i.e. if number of input images is greater than 1
        if len(X.shape) == 3:

            self.n_eval_images = X.shape[0]

            if eval_class_type == 'C2':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X[i,:,:]
                    predicted_img = self.predict(image)[0]

                    label = Y[i,:]
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_2(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    c2_output = self.U_o.dot(predicted_img)
                    guess = softmax(c2_output)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            elif eval_class_type == 'C1':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X[i,:,:]
                    predicted_img = self.predict(image)[0]
                    label = Y[i,:]
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_1(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    guess = softmax(predicted_img)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            else:
                print("classification_type must ='C1' or 'C2'")
                return

            return

        # if X is a single image
        # i.e a vector of shape [:,:].
        elif len(X.shape) == 2:

            self.n_eval_images = 1

            if eval_class_type == 'C2':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X
                    predicted_img = self.predict(image)[0]
                    label = Y
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_2(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    c2_output = self.U_o.dot(predicted_img)
                    guess = softmax(c2_output)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            elif eval_class_type == 'C1':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X
                    predicted_img = self.predict(image)[0]
                    label = Y
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_1(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    guess = softmax(predicted_img)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            else:
                print("classification_type must ='C1' or 'C2'")
                return

        else:
            print("input vector must be 2 or 3-dim")
            return

        print("Evaluation finished.")
        print('\n')

        return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

class TiledPredictiveCodingClassifier:
    def __init__(self, parameters):

        self.is_tiled = True

        self.p = parameters

        # possible choices for transformations, priors
        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}


        # priors and transforms
        self.f = self.unit_act[self.p.unit_act]
        # how to call f(x): self.f(self.U.dot(self.r))[0]
        # how to call F(x): self.f(self.U.dot(self.r))[1]

        self.g = self.prior_dict[self.p.r_prior]
        # how to call g(r): self.g(self.r,self.p.alpha)[0]
        # how to call g'(r): self.g(self.r,self.p.alpha)[1]

        self.h = self.prior_dict[self.p.U_prior]
        # how to call h(U): self.h(self.U,self.p.lam)[0]
        # how to call h'(U): self.h(self.U,self.p.lam)[1]


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


        #make initial learning rate self.variables for plotting.py
        self.lr_r = list(list(self.p.k_r_sched.values())[0].values())[0]
        self.lr_U = list(list(self.p.k_U_sched.values())[0].values())[0]
        self.lr_o = list(list(self.p.k_o_sched.values())[0].values())[0]

        # total number of layers (Input, r1, r2... rn, Output)
        self.n_layers = len(self.p.hidden_sizes) + 2
        # number of non-image layers (r1, r2... rn, Output)
        self.n_non_input_layers = len(self.p.hidden_sizes) + 1
        # number of hidden layers (r1, r2... rn)
        self.n_hidden_layers = len(self.p.hidden_sizes)

        # initialize appropriate r's and U's
        # N.B - MAY NEED TO SET THE FOLLOWING PROPERLY USING PRIORS AT SOME POINT

        # image width/height should be sqrt(576) = 24 pixels
        self.input_image_height = np.sqrt(self.p.input_size)
        self.input_image_width = np.sqrt(self.p.input_size)

        self.r0_single_tile_area = int(self.input_image_height * (self.input_image_width - 2 * self.p.tile_offset))
        print('area of a single tile is {}'.format(self.r0_single_tile_area))
        print('\n')

        # representations

        self.r0 = np.zeros((3,self.r0_single_tile_area,1))
        self.r1 = np.zeros((3,int(self.p.hidden_sizes[0]/3),1))
        # all the representations (including the image r[0] which is not trained)
        print('shape of r0 is ')
        print(self.r0.shape)
        print('shape of r0.0 is')
        print(self.r0[0].shape)
        print('shape of r1 is ')
        print(self.r1.shape)
        # self.r1 will = self.r[1] once filled
        self.r = {}

        # initialize image input layer in three tiles of of size (single tile area,1)
        self.r0[0] = np.random.randn(self.r0_single_tile_area,1)
        self.r0[1] = np.random.randn(self.r0_single_tile_area,1)
        self.r0[2] = np.random.randn(self.r0_single_tile_area,1)

        self.r[0] = np.array([self.r0[0], self.r0[1], self.r0[2]])
        print('shape of r[0] init is {}'.format(self.r[0].shape))

        # print('r0.0 is')
        # print(self.r0[0])

        # initialize first 'true' (non-0) r layer, 3 modules
        # hidden_sizes must be evenly-divisible by 3 for this to work
        # i.e. if hidden_sizes = 96, every r1 module will = size 32,1
        self.r1[0] = np.random.randn(int(self.p.hidden_sizes[0]/3),1)
        self.r1[1] = np.random.randn(int(self.p.hidden_sizes[0]/3),1)
        self.r1[2] = np.random.randn(int(self.p.hidden_sizes[0]/3),1)

        # make all three modules accessible through one primary index (r[1]) for main train() loop
        self.r[1] = np.array([self.r1[0], self.r1[1], self.r1[2]])
        print('shape of r[1] init is {}'.format(self.r[1].shape))

        # initialize first U layer, 3 modules
        # synaptic weights controlling reconstruction in the network
        # U1 is its own list with U1[0] = U1.1, U1[1] = U1.2, and U1[2] = U1.3
        self.U1 = np.zeros((3,self.r0_single_tile_area,int(self.p.hidden_sizes[0]/3)))
        # self.U1 will = self.U[1] once filled
        self.U = {}


        # in 24x24 image example, with 3 tiles, and 6 offset (each tile = 12x24, or 288 pixels), and r1 modules size = 32
        # each U1 module should be size (288 (tile pixels), 32 (number of neurons in each r1 module))
        # could have also initialized them like above: ...randn(self.r0_single_tile_area,int(self.p.hidden_sizes[0]/3))


        self.U1[0] = np.random.randn(len(self.r0[0]),len(self.r1[0]))
        self.U1[1] = np.random.randn(len(self.r0[1]),len(self.r1[1]))
        self.U1[2] = np.random.randn(len(self.r0[2]),len(self.r1[2]))

        # make all three modules accessible through one primary index (U[1]) for main train() loop
        # for a total of (864,32) when stacked vertically (axis=0)
        self.U[1] = np.array([self.U1[0], self.U1[1], self.U1[2]])

        print('shape of U[1] init is {}'.format(self.U[1].shape))

        print('\n')
        print("*** Predictive Coding Classifier ***")
        print('\n')
        print("*** Layer shapes ***")
        print('\n')
        print("r[0] shape is " + str(np.shape(self.r[0])))
        print('\n')

        # initialize r's and U's for hidden layers number 2 to n (i.e. excluding 0th (image) and 1st (tiled) layers)

        for i in range(2,self.n_non_input_layers):
            if i == 2:
                self.r[i] = np.random.randn(self.p.hidden_sizes[i-1],1)
                self.U[i] = np.random.randn(len(self.r[i-1])*self.r[i-1].shape[1],len(self.r[i]))

            elif i >= 3:
                self.r[i] = np.random.randn(self.p.hidden_sizes[i-1],1)
                self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))



        # dict to contain number of neural network parameters per layer comprising the model
        self.nn_parameters_dict = {}

        print('shape of each layer is')

        # fill number of parameters dictionary
        for layer in range(0,self.n_non_input_layers):

            num_r_params = 1
            num_U_params = 1


            if layer < 1:

                for rdim in np.shape(self.r[layer]):

                    num_r_params *= rdim

                ri = "r_{}".format(layer)

                self.nn_parameters_dict[ri] = num_r_params


                print("r[{}] shape is ".format(layer) + str(np.shape(self.r[layer])))

            elif layer == 1:

                for rdim in np.shape(self.r[layer]):

                    num_r_params *= rdim

                for Udim in np.shape(self.U[layer]):

                    num_U_params *= Udim

                ri = "r_{}".format(layer)
                Ui = "U_{}".format(layer)

                self.nn_parameters_dict[ri] = num_r_params
                self.nn_parameters_dict[Ui] = num_U_params

                print("r[{}] shape is ".format(layer) + str(np.shape(self.r[layer])))
                print("U[{}] shape is ".format(layer) + str(np.shape(self.U[layer])))

            elif layer == 2:

                for rdim in np.shape(self.r[layer]):

                    num_r_params *= rdim

                for Udim in np.shape(self.U[layer]):

                    num_U_params *= Udim

                ri = "r_{}".format(layer)
                Ui = "U_{}".format(layer)

                self.nn_parameters_dict[ri] = num_r_params
                self.nn_parameters_dict[Ui] = num_U_params

                print("r[{}] shape is ".format(layer) + str(np.shape(self.r[layer])))
                print("U[{}] shape is ".format(layer) + str(np.shape(self.U[layer])))


            elif layer >= 3:

                for rdim in np.shape(self.r[layer]):

                    num_r_params *= rdim

                for Udim in np.shape(self.U[layer]):

                    num_U_params *= Udim

                ri = "r_{}".format(layer)
                Ui = "U_{}".format(layer)

                self.nn_parameters_dict[ri] = num_r_params
                self.nn_parameters_dict[Ui] = num_U_params

                print("r[{}] shape is ".format(layer) + str(np.shape(self.r[layer])))
                print("U[{}] shape is ".format(layer) + str(np.shape(self.U[layer])))
                print('\n')


            else:
                print("num layers must be natural number")


        print('number of parameters by layer is')
        print(self.nn_parameters_dict)


        # print("len(self.r): " + str(len(self.r)))
        # print('\n')

        # initialize "output" layer
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        # print("o shape is " + str(np.shape(self.o)))
        # print('\n')
        # print("U_o shape is " + str(np.shape(self.U_o)))
        # print('\n')

        # calculate total number of network parameters comprising the model (hidden layers only)
        self.n_model_parameters = sum(self.nn_parameters_dict.values())
        print('total number of model_parameters is {}'.format(self.n_model_parameters))
        print('\n')

        # print("*** Network Parameters ***")
        # print('\n')
        # print("number of neural network parameters per hidden layer: " + str(self.nn_parameters_dict))
        # print('\n')
        # print("total number of hidden layer parameters: {}".format(self.n_model_parameters))
        # print('\n')

        # initialize number of training images, prediction images, and prediction updates
        # will stay 0 if model untrained, or if predict() has not yet been used on it, respectively
        self.n_training_images = 0
        self.n_pred_images = 0
        self.n_pred_updates = 0

        # all below will stay empty lists if model has not been trained,
        # or if predict() has not yet been used on it, respectively

        # average cost per epoch during training; just representation terms
        self.E_avg_per_epoch = []
        # average cost per epoch during training; just classification term
        self.C_avg_per_epoch = []
        # accuracy per epoch during training
        self.acc_per_epoch = []
        # actual prediction vectors (r[n]'s)
        self.prediction = []

        return



    def rep_cost(self):
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        '''
        E = 0

        # when rep_cost is being run, r1 and U1 have already been concatenated in train() (starting directly post-HL1 update)
        # (their concatenated versions are used in HL2 update)
        # therefore we use indexing by thirds instead of referencing 3 separate module objects

        num_modules = self.r[0].shape[0]
        # print('num modules in rep cost is {}'.format(num_modules))
        # LSQ cost

        # normal case
        if len(self.r[0].shape) == 2:
            for i in range(0,len(self.r)-1):
                if i == 0:
                    rthird = int(self.r[i+1].shape[0]/3)
                    Uthird = int(self.U[i+1].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    Uindex1 = 0
                    Uindex2 = Uthird
                    print('shape of r[0] in rep cost LSQ is {}'.format(self.r[0].shape))
                    # in range(0,num_modules)
                    for module in range(0,num_modules):
                        print('shape of r[0][{}] in rep cost LSQ is {}'.format(module, self.r[i][module].shape))
                        print('shape of r[1][{}:{}] in rep cost LSQ is {}'.format(rindex1,rindex2,self.r[1][rindex1:rindex2].shape))
                        print('shape of U[1][{}:{}] in rep cost LSQ is {}'.format(Uindex1,Uindex2,self.U[1][Uindex1:Uindex2].shape))
                        # print('lsq self.U[{}][{}:{}] shape is {}'.format(i+1,Uindex1,Uindex2,self.U[i+1][Uindex1:Uindex2].shape))
                        # print('lsq self.r[{}][{}:{}] shape is {}'.format(i+1,rindex1,rindex2,self.r[i+1][rindex1:rindex2].shape))
                        v = (self.r[i][module] - self.f(self.U[i+1][Uindex1:Uindex2].dot(self.r[i+1][rindex1:rindex2]))[0])
                        E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                        rindex1 += rthird
                        rindex2 += rthird
                        Uindex1 += Uthird
                        Uindex2 += Uthird
                elif i == 1:
                    rthird = int(self.r[i].shape[0]/3)
                    Uthird = int(self.U[i+1].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    Uindex1 = 0
                    Uindex2 = Uthird
                    # in range(0,num_modules)
                    for module in range(0,num_modules):
                        # print('lsq self.U[{}][{}:{}] shape is {}'.format(i+1,Uindex1,Uindex2,self.U[i+1][Uindex1:Uindex2].shape))
                        # print('lsq self.r[{}][{}:{}] shape is {}'.format(i,rindex1,rindex2,self.r[i][rindex1:rindex2].shape))
                        v = (self.r[i][rindex1:rindex2] - self.f(self.U[i+1][Uindex1:Uindex2].dot(self.r[i+1]))[0])
                        E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                        rindex1 += rthird
                        rindex2 += rthird
                        Uindex1 += Uthird
                        Uindex2 += Uthird
                elif i >= 2:
                    v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                    E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                else:
                    print('i must be natural number')

            # priors on r[1],...,r[n]; U[1],...,U[n]
            for i in range(1,len(self.r)):
                if i == 1:
                    rthird = int(self.r[i].shape[0]/3)
                    Uthird = int(self.U[i].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    Uindex1 = 0
                    Uindex2 = Uthird
                    for module in range(0,num_modules):
                        # print('prior self.U[{}][{}:{}] shape is {}'.format(i,Uindex1,Uindex2,self.U[i][Uindex1:Uindex2].shape))
                        # print('prior self.r[{}][{}:{}] shape is {}'.format(i,rindex1,rindex2,self.r[i][rindex1:rindex2].shape))
                        E = E + (self.h(self.U[i][Uindex1:Uindex2],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i][rindex1:rindex2]),self.p.alpha[i])[0])
                        rindex1 += rthird
                        rindex2 += rthird
                        Uindex1 += Uthird
                        Uindex2 += Uthird
                elif i >= 2:
                    E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])
            return E

        # the case when called in evaluate()
        elif len(self.r[0].shape) == 3:
            for i in range(0,len(self.r)-1):
                if i == 0:
                    rthird = int(self.r[i+1].shape[0]/3)
                    # Uthird = int(self.U[i+1].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    # Uindex1 = 0
                    # Uindex2 = Uthird
                    # print('shape of r[0] in rep cost LSQ is {}'.format(self.r[0].shape))
                    # in range(0,num_modules)
                    for module in range(0,num_modules):
                        # print('shape of r[0][{}] in rep cost LSQ is {}'.format(module, self.r[i][module].shape))
                        # print('shape of r[1][{}:{}] in rep cost LSQ is {}'.format(rindex1,rindex2,self.r[1][rindex1:rindex2].shape))
                        # print('Uthird is {}'.format(Uthird))
                        # print('shape of U[1][{}:{}] in rep cost LSQ is {}'.format(Uindex1,Uindex2,self.U[1][Uindex1:Uindex2].shape))
                        # print('lsq self.U[{}][{}:{}] shape is {}'.format(i+1,Uindex1,Uindex2,self.U[i+1][Uindex1:Uindex2].shape))
                        # print('lsq self.r[{}][{}:{}] shape is {}'.format(i+1,rindex1,rindex2,self.r[i+1][rindex1:rindex2].shape))
                        v = (self.r[i][module] - self.f(self.U[i+1][module].dot(self.r[i+1][rindex1:rindex2]))[0])
                        E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                        rindex1 += rthird
                        rindex2 += rthird
                        # Uindex1 += Uthird
                        # Uindex2 += Uthird
                elif i == 1:
                    rthird = int(self.r[i].shape[0]/3)
                    Uthird = int(self.U[i+1].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    Uindex1 = 0
                    Uindex2 = Uthird
                    # in range(0,num_modules)
                    for module in range(0,num_modules):
                        # print('lsq self.U[{}][{}:{}] shape is {}'.format(i+1,Uindex1,Uindex2,self.U[i+1][Uindex1:Uindex2].shape))
                        # print('lsq self.r[{}][{}:{}] shape is {}'.format(i,rindex1,rindex2,self.r[i][rindex1:rindex2].shape))
                        v = (self.r[i][rindex1:rindex2] - self.f(self.U[i+1][Uindex1:Uindex2].dot(self.r[i+1]))[0])
                        E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                        rindex1 += rthird
                        rindex2 += rthird
                        Uindex1 += Uthird
                        Uindex2 += Uthird
                elif i >= 2:
                    v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                    E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
                else:
                    print('i must be natural number')

            # priors on r[1],...,r[n]; U[1],...,U[n]
            for i in range(1,len(self.r)):
                if i == 1:
                    rthird = int(self.r[i].shape[0]/3)
                    Uthird = int(self.U[i].shape[0]/3)
                    rindex1 = 0
                    rindex2 = rthird
                    Uindex1 = 0
                    Uindex2 = Uthird
                    for module in range(0,num_modules):
                        # print('prior self.U[{}][{}:{}] shape is {}'.format(i,Uindex1,Uindex2,self.U[i][Uindex1:Uindex2].shape))
                        # print('prior self.r[{}][{}:{}] shape is {}'.format(i,rindex1,rindex2,self.r[i][rindex1:rindex2].shape))
                        E = E + (self.h(self.U[i][Uindex1:Uindex2],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i][rindex1:rindex2]),self.p.alpha[i])[0])
                        rindex1 += rthird
                        rindex2 += rthird
                        Uindex1 += Uthird
                        Uindex2 += Uthird
                elif i >= 2:
                    E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])
            return E

        else:
            print('len r0.shape needs to be 2 or 3')
            return

    def class_cost_1(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C1. """
        n = self.n_hidden_layers
        C1 = -1*label[None,:].dot(np.log(softmax(self.r[n])))[0,0]
        return C1



    def class_cost_2(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C2, uninclusive of the prior term. """
        n = self.n_hidden_layers
        C2 = -1*label[None,:].dot(np.log(softmax((self.U_o.dot(self.r[n])))))[0,0]
        return C2

    def prediction_error(self,layer_number):
        """ Calculates the normed prediction error of a layer (in [i,n]), i.e. the difference between a
        the layer's image prediction ("r^td") and the image representation from the layer below ("r"). """
        num_modules = self.U[1].shape[0]

        # for case of first predict() update
        if len(self.r[1].shape) == 3:
            if layer_number == 1:
                # print('layer 1')
                # print('r[0] shape is {}'.format(self.r[layer_number-1].shape))
                # print('r[1] shape is {}'.format(self.r[layer_number].shape))
                # print('r[2] shape is {}'.format(self.r[layer_number+1].shape))
                # print('U[1] shape is {}'.format(self.U[layer_number].shape))
                # print('U[2] shape is {}'.format(self.U[layer_number+1].shape))
                pe = 0
                # rthird = int(self.r[layer_number].shape[0]/3)
                # rindex1 = 0
                # rindex2 = rthird
                for mod in range(0,num_modules):
                    # print('r[0][{}] shape is {}'.format(mod,self.r[layer_number-1][mod].shape))
                    # print('r[1][{}:{}] shape is {}'.format(rindex1,rindex2,self.r[layer_number][rindex1:rindex2].shape))
                    # print()
                    pe = pe + math.sqrt((self.r[layer_number-1][mod]-self.f(self.U[layer_number][mod].dot(self.r[layer_number][mod]))[0]).T\
                    .dot(self.r[layer_number-1][mod]-self.f(self.U[layer_number][mod].dot(self.r[layer_number][mod]))[0]))
                    # rindex1 += rthird
                    # rindex2 += rthird
                return pe

            elif layer_number == 2:

                # print('layer 2')
                # print('r[0] shape is {}'.format(self.r[layer_number-2].shape))
                # print('r[1] shape is {}'.format(self.r[layer_number-1].shape))
                # print('r[2] shape is {}'.format(self.r[layer_number].shape))
                # print('U[1] shape is {}'.format(self.U[layer_number-1].shape))
                # print('U[2] shape is {}'.format(self.U[layer_number].shape))
                pe = 0

                # Uthird1 = int(self.U[layer_number-1].shape[0]/3)
                # rindex1_1 = 0
                # rindex1_2 = rthird1
                # rthird2 = int(self.r[layer_number].shape[0]/3)
                # rindex2_1 = 0
                # rindex2_2 = rthird2

                # concatenate r1
                self.r[layer_number-1] = np.concatenate((self.r[layer_number-1][0], self.r[layer_number-1][1], self.r[layer_number-1][2]),axis=None)[:,None]
                # print("self.r[1] shape after concat is {}".format(self.r[1].shape))

                for mod in range(0,num_modules):
                    pe = pe + math.sqrt((self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]).T\
                    .dot(self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]))

                #split r[1] back up into an array of (1/3rd-sized) arrays

                # print("self.r[1] shape before splitting is {}".format(self.r[1].shape))

                rthird = int(self.r[1].shape[0]/3)
                # print('rthird is {}'.format(rthird))
                rindex1 = 0
                rindex2 = rindex1 + rthird
                rindex3 = rindex2 + rthird
                rindex4 = rindex3 + rthird

                self.r[1] = np.array([self.r[1][rindex1:rindex2],self.r[1][rindex2:rindex3],self.r[1][rindex3:rindex4]])
                # print("self.r[1] shape after splitting is {}".format(self.r[1].shape))

                return pe
            else:
                print('layer_number must be 1 or 2')
                return

        # for the cases of all subsequent predict() updates where r1 has been concatenated
        elif len(self.r[1].shape) == 2:
            if layer_number == 1:
                # print('layer 1')
                # print('r[0] shape is {}'.format(self.r[layer_number-1].shape))
                # print('r[1] shape is {}'.format(self.r[layer_number].shape))
                # print('r[2] shape is {}'.format(self.r[layer_number+1].shape))
                # print('U[1] shape is {}'.format(self.U[layer_number].shape))
                # print('U[2] shape is {}'.format(self.U[layer_number+1].shape))

                # print("self.r[1] shape before splitting is {}".format(self.r[1].shape))

                rthird = int(self.r[1].shape[0]/3)
                # print('rthird is {}'.format(rthird))
                rindex1 = 0
                rindex2 = rindex1 + rthird
                rindex3 = rindex2 + rthird
                rindex4 = rindex3 + rthird

                self.r[1] = np.array([self.r[1][rindex1:rindex2],self.r[1][rindex2:rindex3],self.r[1][rindex3:rindex4]])
                # print("self.r[1] shape after splitting is {}".format(self.r[1].shape))

                pe = 0
                # rthird = int(self.r[layer_number].shape[0]/3)
                # rindex1 = 0
                # rindex2 = rthird
                for mod in range(0,num_modules):
                    # print('r[0][{}] shape is {}'.format(mod,self.r[layer_number-1][mod].shape))
                    # print('r[1][{}:{}] shape is {}'.format(rindex1,rindex2,self.r[layer_number][rindex1:rindex2].shape))
                    # print()
                    pe = pe + math.sqrt((self.r[layer_number-1][mod]-self.f(self.U[layer_number][mod].dot(self.r[layer_number][mod]))[0]).T\
                    .dot(self.r[layer_number-1][mod]-self.f(self.U[layer_number][mod].dot(self.r[layer_number][mod]))[0]))
                    # rindex1 += rthird
                    # rindex2 += rthird
                return pe

            elif layer_number == 2:

                # print('layer 2')
                # print('r[0] shape is {}'.format(self.r[layer_number-2].shape))
                # print('r[1] shape is {}'.format(self.r[layer_number-1].shape))
                # print('r[2] shape is {}'.format(self.r[layer_number].shape))
                # print('U[1] shape is {}'.format(self.U[layer_number-1].shape))
                # print('U[2] shape is {}'.format(self.U[layer_number].shape))
                pe = 0

                # Uthird1 = int(self.U[layer_number-1].shape[0]/3)
                # rindex1_1 = 0
                # rindex1_2 = rthird1
                # rthird2 = int(self.r[layer_number].shape[0]/3)
                # rindex2_1 = 0
                # rindex2_2 = rthird2

                # concatenate r1
                self.r[layer_number-1] = np.concatenate((self.r[layer_number-1][0], self.r[layer_number-1][1], self.r[layer_number-1][2]),axis=None)[:,None]
                # print("self.r[1] shape after concat is {}".format(self.r[1].shape))

                for mod in range(0,num_modules):
                    pe = pe + math.sqrt((self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]).T\
                    .dot(self.r[layer_number-1]-self.f(self.U[layer_number].dot(self.r[layer_number]))[0]))

                #split r[1] back up into an array of (1/3rd-sized) arrays

                # print("self.r[1] shape before splitting is {}".format(self.r[1].shape))

                rthird = int(self.r[1].shape[0]/3)
                # print('rthird is {}'.format(rthird))
                rindex1 = 0
                rindex2 = rindex1 + rthird
                rindex3 = rindex2 + rthird
                rindex4 = rindex3 + rthird

                self.r[1] = np.array([self.r[1][rindex1:rindex2],self.r[1][rindex2:rindex3],self.r[1][rindex3:rindex4]])
                # print("self.r[1] shape after splitting is {}".format(self.r[1].shape))

                return pe
            else:
                print('layer_number must be 1 or 2')
                return

        else:
            print('len shape of r1 should be 2 or 3')
            return

    def train(self,X,Y):
        '''
        X: matrix of input patterns (N_patterns x input_size)
        Y: matrix of output/target patterns (N_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        # number of hidden layers
        n = self.n_hidden_layers

        # initialize "output" layer o (for classification method 2 (C2))
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights U_o to the output (C2)
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        """
        FIX n > 2
        later
        2021.06.10
        """

        if n > 2:

            print("*** Training ***")
            print('\n')

            # loop through training image dataset num_epochs times
            for epoch in range(0,self.p.num_epochs):
                # shuffle order of training set input image / output vector pairs each epoch
                N_permuted_indices = np.random.permutation(X.shape[0])
                X_shuffled = X[N_permuted_indices]
                Y_shuffled = Y[N_permuted_indices]

                # print("y_shuffled shape is: " + '\n' + str(Y_shuffled.shape))

                # number of training images
                self.n_training_images = X_shuffled.shape[0]

                # we compute average cost per epoch (batch size = 1); separate classification
                #   and representation costs so we can compare OOM sizes
                E = 0
                C = 0

                # accuracy per epoch: how many images are correctly guessed per epoch
                self.num_correct = 0

                # set learning rates at the start of each epoch
                k_r = self.k_r_lr(epoch)
                k_U = self.k_U_lr(epoch)
                k_o = self.k_o_lr(epoch)

                # print("*** train() function values and layers ***")
                # print("Number of training images is {}".format(num_images) + '\n')

                print("epoch {}".format(epoch+1))

                # loop through training images
                for image in range(0, self.n_training_images):


                    # copy image tiles into r[0]
                    # turn (576,) image into (1,576) and inflate to (1,24,24)
                    image_expanded = data.inflate_vectors(X_shuffled[image,:][None,:])
                    print('image expanded shape is {}'.format(image_expanded.shape))
                    image_squeezed = np.squeeze(image_expanded)
                    print('image squeezed shape is {}'.format(image_squeezed.shape))
                    cut_image = data.cut(image_squeezed,tile_offset=6)
                    print('cut image tuple length is {}'.format(len(cut_image)))
                    print('cut image[0] shape is {}'.format(cut_image[0].shape))
                    print('squeezed cut image[0] shape is {}'.format(np.squeeze(cut_image[0]).shape))
                    squeezed_tile1 = np.squeeze(cut_image[0])
                    squeezed_tile2 = np.squeeze(cut_image[1])
                    squeezed_tile3 = np.squeeze(cut_image[2])

                    self.r[0][0] = squeezed_tile1[:,None]
                    self.r[0][1] = squeezed_tile2[:,None]
                    self.r[0][2] = squeezed_tile3[:,None]


                    # initialize new r's
                    for layer in range(1,self.n_non_input_layers):
                        # self state per layer
                        self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)

                    # designate label vector
                    label = Y_shuffled[image,:]




                    # loop through intermediate layers (will fail if number of hidden layers is 1)
                    # r,U updates written symmetrically for all layers including output


                    for i in range(2,n):


                        # NOTE: self.p.k_r learning rate
                        # r[i] update
                        self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                        * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                        + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                        - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]


                        # U[i] update
                        self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                        * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                        - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]


                    """ r(n) update (C1) """
                    self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
                    * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                    - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \
                    # # classification term
                    + (k_o / 2) * (label[:,None] - softmax(self.r[n]))

                    # U[n] update (C1, C2) (identical to U[i], except index numbers)
                    self.U[n] = self.U[n] + (k_U / self.p.sigma_sq[n]) \
                    * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                    - (k_U / 2) * self.h(self.U[n],self.p.lam[n])[1]

                    # Loss function E
                    E = E + self.rep_cost()


                    # Classification cost function C

                    # if not classifying, default classification type is "NC"
                    self.class_type = 'NC'


                    """ Classifying using C1 """
                    C = self.class_cost_1(label)
                    E = E + C
                    self.class_type = 'C1'

                    """ C1 method """
                    if np.argmax(softmax(self.r[n])) == np.argmax(label[:,None]):
                        self.num_correct += 1




                # store average costs and accuracy per epoch
                E_avg_per_epoch = E/self.n_training_images
                C_avg_per_epoch = C/self.n_training_images
                acc_per_epoch = round((self.num_correct/self.n_training_images)*100)

                self.E_avg_per_epoch.append(E_avg_per_epoch)
                self.C_avg_per_epoch.append(C_avg_per_epoch)
                self.acc_per_epoch.append(acc_per_epoch)


                """
                +++
                ***
                !!!
                STUFF BELOW THIS IS
                HL (n) <= 2
                !!!
                ***
                +++
                """


        elif n <= 2:

            print("*** Training ***")
            print('\n')

            # loop through training image dataset num_epochs times
            for epoch in range(0,self.p.num_epochs):
                # shuffle order of training set input image / output vector pairs each epoch
                N_permuted_indices = np.random.permutation(X.shape[0])
                X_shuffled = X[N_permuted_indices]
                Y_shuffled = Y[N_permuted_indices]

                # print("y_shuffled shape is: " + '\n' + str(Y_shuffled.shape))

                # number of training images
                self.n_training_images = X_shuffled.shape[0]

                # we compute average cost per epoch (batch size = 1); separate classification
                #   and representation costs so we can compare OOM sizes
                E = 0
                C = 0

                # accuracy per epoch: how many images are correctly guessed per epoch
                self.num_correct = 0

                # set learning rates at the start of each epoch
                k_r = self.k_r_lr(epoch)
                k_U = self.k_U_lr(epoch)
                k_o = self.k_o_lr(epoch)

                # print("*** train() function values and layers ***")
                # print("Number of training images is {}".format(num_images) + '\n')

                print("epoch {}".format(epoch+1))

                # loop through training images
                for image in range(0, self.n_training_images):

                    # print('\n')
                    # print('image {}'.format(image+1))
                    # print('\n')

                    # copy image tiles into r[0]
                    # turn (576,) image into (1,576) and inflate to (1,24,24)
                    image_expanded = data.inflate_vectors(X_shuffled[image,:][None,:])


                    # print('image expanded shape is {}'.format(image_expanded.shape))
                    image_squeezed = np.squeeze(image_expanded)

                    # plt.imshow(image_squeezed, cmap='Greys'),plt.title('image #{} post inflation'.format(image+1))
                    # plt.show()

                    # print('image squeezed shape is {}'.format(image_squeezed.shape))
                    cut_image = data.cut(image_squeezed,tile_offset=6,flat=True)

                    # print('cut image tuple length is {}'.format(len(cut_image)))
                    # print('cut image[0] shape is {}'.format(cut_image[0].shape))
                    # print('squeezed cut image[0] shape is {}'.format(np.squeeze(cut_image[0]).shape))
                    # print('\n')

                    squeezed_tile1 = np.squeeze(cut_image[0])
                    squeezed_tile2 = np.squeeze(cut_image[1])
                    squeezed_tile3 = np.squeeze(cut_image[2])

                    # tile1inf = data.inflate_vectors(cut_image[0],shape_2d=(24,12))
                    # tile2inf = data.inflate_vectors(cut_image[1],shape_2d=(24,12))
                    # tile3inf = data.inflate_vectors(cut_image[2],shape_2d=(24,12))

                    # tile1img = np.squeeze(tile1inf)
                    # tile2img = np.squeeze(tile2inf)
                    # tile3img = np.squeeze(tile3inf)

                    # # plot
                    # plt.subplot(131),plt.imshow(tile1img, cmap='Greys'),plt.title('image {} tile #1'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.subplot(132),plt.imshow(tile2img, cmap='Greys'),plt.title('image {} tile #2'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.subplot(133),plt.imshow(tile3img, cmap='Greys'),plt.title('image {} tile #3'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.show()

                    # # plot
                    # plt.subplot(131),plt.imshow(cut_image[0], cmap='Greys'),plt.title('image {} tile #1'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.subplot(132),plt.imshow(cut_image[1], cmap='Greys'),plt.title('image {} tile #2'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.subplot(133),plt.imshow(cut_image[2], cmap='Greys'),plt.title('image {} tile #3'.format(image+1))
                    # plt.xticks([]), plt.yticks([])
                    # # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.show()

                    self.r[0][0] = squeezed_tile1[:,None]
                    self.r[0][1] = squeezed_tile2[:,None]
                    self.r[0][2] = squeezed_tile3[:,None]

                    # print('shape of r[0][0] image loop is {}'.format(self.r[0][0].shape))
                    # print('shape of r[0][1] image loop is {}'.format(self.r[0][1].shape))
                    # print('shape of r[0][2] image loop is {}'.format(self.r[0][2].shape))
                    # print('shape of r[0] image loop is {}'.format(self.r[0].shape))
                    # print('\n')

                    # print('shape of r[1][0] image loop is {}'.format(self.r[1][0].shape))
                    # print('shape of r[1][1] image loop is {}'.format(self.r[1][1].shape))
                    # print('shape of r[1][2] image loop is {}'.format(self.r[1][2].shape))
                    # print('shape of r[1] image loop is {}'.format(self.r[1].shape))
                    # print('\n')

                    # print('shape of r[2] image loop is {}'.format(self.r[2].shape))
                    # print('\n')

                    # print('shape of U[1][0] image loop is {}'.format(self.U[1][0].shape))
                    # print('shape of U[1][1] image loop is {}'.format(self.U[1][1].shape))
                    # print('shape of U[1][2] image loop is {}'.format(self.U[1][2].shape))
                    # print('shape of U[1] image loop is {}'.format(self.U[1].shape))
                    # print('\n')

                    # print('shape of U[2][0] image loop is {}'.format(self.U[2][0].shape))
                    # print('shape of U[2] image loop is {}'.format(self.U[2].shape))
                    # print('\n')

                    # initialize new r's
                    for layer in range(1,self.n_non_input_layers):
                        if layer == 1:
                            self.r[layer] = np.zeros((3,int(self.p.hidden_sizes[0]/3),1))
                            for module in range(0,self.r[1].shape[0]):
                                # self state per layer
                                self.r[layer][module] = np.random.randn(int(self.p.hidden_sizes[layer-1]/3),1)
                                # print('shape of reinitialized r[{}][{}] is {}'.format(layer, module, self.r[layer][module].shape))
                            # print('shape of total reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))
                        elif layer >= 2:
                            # self state per layer
                            self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                            # print('shape of reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))

                        else:
                            print('new rs in every image start numbered at 1 and go to n. num layers must be natural number')


                    # designate label vector
                    label = Y_shuffled[image,:]


                    U2third = int(self.U[2].shape[0]/3)
                    U2index1 = 0
                    U2index2 = U2third

                    # update r/U in each module and each layer

                    num_modules = self.r[0].shape[0]

                    for i in range(0,num_modules):

                        # update r1.1-3 and U1.1-3 before moving on to r2/U2 (this is HL1 in the 2HL model special case)

                        # print('\n')
                        # print("start module {} update".format(i+1))
                        # print('\n')

                        # print('shape of r[0][{}] update loop pre-update is {}'.format(i, self.r[0][i].shape))
                        # print('shape of r[1][{}] update loop pre-update is {}'.format(i, self.r[1][i].shape))
                        # print('shape of r[2] update loop pre-update is {}'.format(self.r[2].shape))
                        # print('shape of U[1][{}] update loop pre-update is {}'.format(i, self.U[1][i].shape))
                        # print('shape of U[2] update loop pre-update is {}'.format(self.U[2].shape))
                        # print('shape of U[2][{}] update loop pre-update is {}'.format(i, self.U[2][i].shape))

                        # print('self.p.alpha[1] is {}'.format(self.p.alpha[1]))

                        # NOTE: self.p.k_r learning rate
                        # r[i] update
                        self.r[1][i] = self.r[1][i] + (k_r / self.p.sigma_sq[1]) \
                        * self.U[1][i].T.dot(self.f(self.U[1][i].dot(self.r[1][i]))[1].dot(self.r[0][i] - self.f(self.U[1][i].dot(self.r[1][i]))[0])) \
                        + (k_r / self.p.sigma_sq[2]) * (self.f(self.U[2][U2index1:U2index2].dot(self.r[2]))[0] - self.r[1][i]) \
                        - (k_r / 2) * self.g(self.r[1][i],self.p.alpha[1])[1]

                        # print('shape of U[2][{}:{}] update loop pre-update is {}'.format(U2index1,U2index2,self.U[2][U2index1:U2index2].shape))


                        U2index1 += U2third
                        U2index2 += U2third

                        # print('shape of U[2] update loop post-update is {}'.format(self.U[2].shape))


                        # U[i] update
                        self.U[1][i] = self.U[1][i] + (k_U / self.p.sigma_sq[1]) \
                        * (self.f(self.U[1][i].dot(self.r[1][i]))[1].dot(self.r[0][i] - self.f(self.U[1][i].dot(self.r[1][i]))[0])).dot(self.r[1][i].T) \
                        - (k_U / 2) * self.h(self.U[1][i],self.p.lam[1])[1]

                    # concatenate r1
                    self.r[1] = np.concatenate((self.r[1][0], self.r[1][1], self.r[1][2]),axis=None)[:,None]
                    # print("self.r[1] shape after concat is {}".format(self.r[1].shape))

                    # concatenated U1 should be 864,32 (stacked vertically)
                    self.U[1] = np.concatenate((self.U[1][0],self.U[1][1],self.U[1][2]),axis=0)

                    # print("self.U[1] shape after concat is {}".format(self.U[1].shape))

                    """ r(n) update (C1) """
                    self.r[2] = self.r[2] + (k_r / self.p.sigma_sq[2]) \
                    * self.U[2].T.dot(self.f(self.U[2].dot(self.r[2]))[1].dot(self.r[1] - self.f(self.U[2].dot(self.r[2]))[0])) \
                    - (k_r / 2) * self.g(self.r[2],self.p.alpha[2])[1] \
                    # classification term
                    + (k_o / 2) * (label[:,None] - softmax(self.r[2]))


                    # U[n] update (C1, C2) (identical to U[i], except index numbers)
                    self.U[2] = self.U[2] + (k_U / self.p.sigma_sq[2]) \
                    * (self.f(self.U[2].dot(self.r[2]))[1].dot(self.r[1] - self.f(self.U[2].dot(self.r[2]))[0])).dot(self.r[2].T) \
                    - (k_U / 2) * self.h(self.U[2],self.p.lam[2])[1]

                    # Loss function E
                    E = E + self.rep_cost()


                    # Classification cost function C

                    # if not classifying, default classification type is "NC"
                    self.class_type = 'NC'


                    """ Classifying using C1 """
                    C = self.class_cost_1(label)
                    E = E + C
                    self.class_type = 'C1'


                    # Classification Accuracy

                    # if index of the max value in final representation vector
                    # = index of the 1 in associated one hot label vector
                    # then the model classified the image correctly
                    # and add 1 to num_correct


                    """ C1 method """
                    if np.argmax(softmax(self.r[n])) == np.argmax(label[:,None]):
                        self.num_correct += 1


                    # print('np.argmax(softmax(c2_output))')
                    # print(np.argmax(softmax(c2_output)))
                    # print('np.argmax(label)')
                    # print(np.argmax(label[:,None]))

                    # print('n_training_images')
                    # print(self.n_training_images)

                    #split U[1] back up into an array of (1/3rd-sized) arrays

                    Uthird = int(self.U[1].shape[0]/3)
                    Uindex1 = 0
                    Uindex2 = Uindex1 + Uthird
                    Uindex3 = Uindex2 + Uthird
                    Uindex4 = Uindex3 + Uthird

                    self.U[1] = np.array([self.U[1][Uindex1:Uindex2],self.U[1][Uindex2:Uindex3],self.U[1][Uindex3:Uindex4]])
                    # print("self.U[1] shape after splitting is {}".format(self.U[1].shape))


                print('self.num_correct epoch {}'.format(epoch+1))
                print(self.num_correct)

                # store average costs and accuracy per epoch
                E_avg_per_epoch = E/self.n_training_images
                C_avg_per_epoch = C/self.n_training_images
                acc_per_epoch = (self.num_correct/self.n_training_images)*100

                self.E_avg_per_epoch.append(E_avg_per_epoch)
                self.C_avg_per_epoch.append(C_avg_per_epoch)
                self.acc_per_epoch.append(acc_per_epoch)

        # else:
        #     print('number of hidden layers must be a natural number')

        return


    def predict(self,X):
        '''
        Given one or more inputs, produce one or more outputs. X should be a matrix of shape [n_pred_images,:,:]
        or a single image of size [:,:]. Predict returns a list of predictions (self.prediction), i.e.
        self.prediction = [[contents_of_r[n]_img1],[contents_of_r[n]_img2]]. Therefore,
        self.prediction[0] = the actual vector of interest (what the model "sees") = [contents_of_r[n]_img1]
        predict() also saves a list of per-update-PEs for each image, split by layer.
        If you predict (e.g.) 2 images, accessing these PEs is as follows:
        image1_layer1PE, image1_layer2PE = self.prediction_errors_l1[0], self.prediction_errors_l2[0]
        image2_layer1PE, image2_layer2PE = self.prediction_errors_l1[1], self.prediction_errors_l2[1]
        '''

        # number of hidden layers
        n = self.n_hidden_layers

        # number of r updates before r's "relax" into a stable representation of the image
        # empirically, between 50-100 seem to work, so we'll stick with 100.

        self.n_pred_updates = 100
        # re-initialize lists of actual prediction outputs and PEs
        self.r1s = []
        self.prediction = []
        self.prediction_errors_l1 = []
        self.prediction_errors_l2 = []

        # set learning rate for r
        k_r = 0.05


        # if X is a matrix of shape [n_pred_images,:,:].
        # i.e if the input is multiple images
        if len(X.shape) == 3:

            print("predicting a 3-dimensional (multi-image) vector")

            self.n_pred_images = X.shape[0]
            print("npredimages")
            print(self.n_pred_images)

            # get from [n,24,24] input to [n,576] so that self.r[0] instantiation below
            # can convert to and call each image as a [576,1] vector
            X_flat = data.flatten_images(X)


            print("*** Predicting ***")
            print('\n')

            # loop through testing images
            for image in range(0, self.n_pred_images):

                print("starting image {}".format(image+1))

                # representation costs for zeroth and nth layers
                self.pe_1 = []
                self.pe_2 = []

                # copy first image into r[0]

                # print(X[image].shape)
                # convert [1,576] image to one [576,1] image
                self.r[0] = X_flat[image,:][:,None]

                # print(X[image].shape)
                # print(X[image][:,None].shape)
                # print(self.r[0].shape)
                # print("fUr shape")
                # print((self.f(self.U[1].dot(self.r[1]))[0]).shape)


                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    if layer == 1:
                        self.r[layer] = np.zeros((3,int(self.p.hidden_sizes[0]/3),1))
                        for module in range(0,self.r[1].shape[0]):
                            # self state per layer
                            self.r[layer][module] = np.random.randn(int(self.p.hidden_sizes[layer-1]/3),1)
                            # print('shape of reinitialized r[{}][{}] is {}'.format(layer, module, self.r[layer][module].shape))
                        # print('shape of total reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))
                    elif layer >= 2:
                        # self state per layer
                        self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                        # print('shape of reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))

                    else:
                        print('new rs in every image start numbered at 1 and go to n. num layers must be natural number')



                for update in range(0,self.n_pred_updates):

                    # print('prediction update {}'.format(update+1))
                    # magnitude (normed) prediction errors each "layer" (i.e. error between r0,r1, and r1,r2)

                    pe_1 = self.prediction_error(1)
                    pe_2 = self.prediction_error(2)

                    self.pe_1.append(pe_1)
                    self.pe_2.append(pe_2)


                    # update r/U in each module and each layer

                    num_modules = self.r[1].shape[0]
                    # print('num_modules is {}'.format(num_modules))

                    U2third = int(self.U[2].shape[0]/3)
                    U2index1 = 0
                    U2index2 = U2third


                    for i in range(0,num_modules):

                        # r[i] update

                        # print('self.r[1] shape is {}'.format(self.r[1].shape))
                        self.r[1][i] = self.r[1][i] + (k_r / self.p.sigma_sq[1]) \
                        * self.U[1][i].T.dot(self.f(self.U[1][i].dot(self.r[1][i]))[1].dot(self.r[0][i] - self.f(self.U[1][i].dot(self.r[1][i]))[0])) \
                        + (k_r / self.p.sigma_sq[2]) * (self.f(self.U[2][U2index1:U2index2].dot(self.r[2]))[0] - self.r[1][i]) \
                        - (k_r / 2) * self.g(self.r[1][i],self.p.alpha[1])[1]

                        U2index1 += U2third
                        U2index2 += U2third

                    # concatenate r1
                    self.r[1] = np.concatenate((self.r[1][0], self.r[1][1], self.r[1][2]),axis=None)[:,None]
                    # print("self.r[1] shape after concat is {}".format(self.r[1].shape))


                    self.r[2] = self.r[2] + (k_r / self.p.sigma_sq[2]) \
                    * self.U[2].T.dot(self.f(self.U[2].dot(self.r[2]))[1].dot(self.r[1] - self.f(self.U[2].dot(self.r[2]))[0])) \
                    - (k_r / 2) * self.g(self.r[2],self.p.alpha[2])[1]


                # return final predictions
                # i.e. r[n]'s

                r1 = self.r[1]
                print('first five lines of r1')
                print(r1[:5])
                prediction = self.r[n]
                print('first five lines of prediction (r2)')
                print(prediction[:5])

                self.r1s.append(r1)
                self.prediction.append(prediction)
                self.prediction_errors_l1.append(self.pe_1)
                self.prediction_errors_l2.append(self.pe_2)


        # if X is a single image
        # of shape [:,:]
        elif len(X.shape) == 2:

            print("predicting a 2-dimensional (single-image) vector")
            # print("Xshape is")
            # print(X.shape)

            self.n_pred_images = 1

            # get from [24,24] input to [1,576] so that self.r[0] instantiation below
            # can convert to and call the image as a [576,1] vector
            X_flat = data.flatten_images(X[None,:,:])

            # print("Xflat is")
            # print(X_flat.shape)

            print("*** Predicting ***")
            print('\n')

            # loop through testing images
            for image in range(0, self.n_pred_images):

                # representation costs for zeroth and nth layers
                self.pe_1 = []
                self.pe_2 = []

                # copy image tiles into r[0]
                # turn (576,) image into (1,576) and inflate to (1,24,24)
                image_expanded = data.inflate_vectors(X_flat)

                image_squeezed = np.squeeze(image_expanded)

                cut_image = data.cut(image_squeezed,tile_offset=6,flat=True)

                squeezed_tile1 = np.squeeze(cut_image[0])
                squeezed_tile2 = np.squeeze(cut_image[1])
                squeezed_tile3 = np.squeeze(cut_image[2])

                self.r[0][0] = squeezed_tile1[:,None]
                self.r[0][1] = squeezed_tile2[:,None]
                self.r[0][2] = squeezed_tile3[:,None]

                # print(X[image].shape)
                # print(X[image][:,None].shape)
                # print(self.r[0].shape)
                # print("fUr shape")
                # print((self.f(self.U[1].dot(self.r[1]))[0]).shape)


                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    if layer == 1:
                        self.r[layer] = np.zeros((3,int(self.p.hidden_sizes[0]/3),1))
                        for module in range(0,self.r[1].shape[0]):
                            # self state per layer
                            self.r[layer][module] = np.random.randn(int(self.p.hidden_sizes[layer-1]/3),1)
                            # print('shape of reinitialized r[{}][{}] is {}'.format(layer, module, self.r[layer][module].shape))
                        # print('shape of total reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))
                    elif layer >= 2:
                        # self state per layer
                        self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)
                        # print('shape of reinitialized r[{}] is {}'.format(layer, self.r[layer].shape))

                    else:
                        print('new rs in every image start numbered at 1 and go to n. num layers must be natural number')

                # print('re initialized self.r[1] shape is {}'.format(self.r[1].shape))

                # print('\n')
                # print('STARTING UPDATES')
                # print('\n')

                for update in range(0,self.n_pred_updates):

                    # print('prediction update {}'.format(update+1))

                    # magnitude (normed) prediction errors each "layer" (i.e. error between r0,r1, and r1,r2)

                    pe_1 = self.prediction_error(1)
                    pe_2 = self.prediction_error(2)

                    self.pe_1.append(pe_1)
                    self.pe_2.append(pe_2)

                    # update r/U in each module and each layer

                    num_modules = self.r[0].shape[0]

                    U2third = int(self.U[2].shape[0]/3)
                    U2index1 = 0
                    U2index2 = U2third

                    # print('after prediction error calcs')
                    # print('r[0] shape is {}'.format(self.r[0].shape))
                    # print('r[1] shape is {}'.format(self.r[1].shape))
                    # print('r[2] shape is {}'.format(self.r[2].shape))
                    # print('U[1] shape is {}'.format(self.U[1].shape))
                    # print('U[2] shape is {}'.format(self.U[2].shape))

                    for i in range(0,num_modules):

                        # print('self.r[1] shape is {}'.format(self.r[1].shape))
                        # print('self.r[1][{}] shape is {}'.format(i, self.r[1][i].shape))

                        # r[i] update
                        self.r[1][i] = self.r[1][i] + (k_r / self.p.sigma_sq[1]) \
                        * self.U[1][i].T.dot(self.f(self.U[1][i].dot(self.r[1][i]))[1].dot(self.r[0][i] - self.f(self.U[1][i].dot(self.r[1][i]))[0])) \
                        + (k_r / self.p.sigma_sq[2]) * (self.f(self.U[2][U2index1:U2index2].dot(self.r[2]))[0] - self.r[1][i]) \
                        - (k_r / 2) * self.g(self.r[1][i],self.p.alpha[1])[1]

                        U2index1 += U2third
                        U2index2 += U2third

                    # concatenate r1
                    self.r[1] = np.concatenate((self.r[1][0], self.r[1][1], self.r[1][2]),axis=None)[:,None]
                    # print("self.r[1] shape after concat is {}".format(self.r[1].shape))


                    self.r[2] = self.r[2] + (k_r / self.p.sigma_sq[2]) \
                    * self.U[2].T.dot(self.f(self.U[2].dot(self.r[2]))[1].dot(self.r[1] - self.f(self.U[2].dot(self.r[2]))[0])) \
                    - (k_r / 2) * self.g(self.r[2],self.p.alpha[2])[1]

                # return final prediction (r[n]) and final r[1]

                r1 = self.r[1]
                prediction = self.r[n]

                self.r1s.append(r1)
                self.prediction.append(prediction)
                self.prediction_errors_l1.append(self.pe_1)
                self.prediction_errors_l2.append(self.pe_2)

        else:
            print("input vector must be 2 or 3-dim")

        return self.prediction

    def evaluate(self,X,Y,eval_class_type='C1'):

        """ evaluates model's E, C and classification accuracy in any state (trained, untrained)
        using any input data. X should be a matrix of shape [n_pred_images,:,:] or a single image of size [:,:]
        Calls self.predict(): predict can take a 3-dim (multi-image) or 2-dim (single image) vector, but when called
        here in evalute(), predict only takes in one image (2-dim vec) at a time."""

        self.E_per_image = []
        self.C_per_image  = []
        self.Classif_success_by_img = []
        self.acc_evaluation = 0
        self.eval_class_type = eval_class_type
        self.softmax_guess_each_img = []

        # if X is a matrix of shape [n_eval_images,:,:].
        # i.e. if number of input images is greater than 1
        if len(X.shape) == 3:

            self.n_eval_images = X.shape[0]

            if eval_class_type == 'C2':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X[i,:,:]
                    predicted_img = self.predict(image)[0]

                    label = Y[i,:]
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_2(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    c2_output = self.U_o.dot(predicted_img)
                    guess = softmax(c2_output)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            elif eval_class_type == 'C1':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X[i,:,:]
                    predicted_img = self.predict(image)[0]
                    label = Y[i,:]
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_1(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    guess = softmax(predicted_img)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            else:
                print("classification_type must ='C1' or 'C2'")
                return

            return

        # if X is a single image
        # i.e a vector of shape [:,:].
        elif len(X.shape) == 2:

            self.n_eval_images = 1

            if eval_class_type == 'C2':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X
                    predicted_img = self.predict(image)[0]
                    label = Y
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_2(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    c2_output = self.U_o.dot(predicted_img)
                    guess = softmax(c2_output)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            elif eval_class_type == 'C1':
                for i in range(0,self.n_eval_images):
                    print("eval image{}".format(i+1))
                    image = X
                    predicted_img = self.predict(image)[0]
                    label = Y
                    Eimg = self.rep_cost()
                    Cimg = self.class_cost_1(label)
                    self.C_per_image.append(Cimg)
                    Eimg = Eimg + Cimg
                    self.E_per_image.append(Eimg)
                    guess = softmax(predicted_img)
                    if np.argmax(guess) == np.argmax(label[:,None]):
                        self.Classif_success_by_img.append(1)
                    else:
                        self.Classif_success_by_img.append(0)
                    self.softmax_guess_each_img.append(guess)
                num_correct = sum(self.Classif_success_by_img)
                self.acc_evaluation = (num_correct / self.n_eval_images) * 100
                return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img

            else:
                print("classification_type must ='C1' or 'C2'")
                return

        else:
            print("input vector must be 2 or 3-dim")
            return

        print("Evaluation finished.")
        print('\n')

        return self.E_per_image,self.C_per_image,self.Classif_success_by_img,self.acc_evaluation,self.softmax_guess_each_img
