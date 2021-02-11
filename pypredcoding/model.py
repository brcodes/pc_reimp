# Implementation of r&b 1999 predictive coding model with MNIST data.

import numpy as np
import pypredcoding.parameters as parameters
import pypredcoding.data as data
import unittest

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
    F = np.diag(1 - np.square(np.tanh(U_dot_r)))
    return (f, F)

# r, U prior functions
def gauss_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior. """
    g_or_h = alph_or_lam * np.square(r_or_U)
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)

def kurt_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior. """
    g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U))
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    return (g_or_h, gprime_or_hprime)

# softmax function
def softmax(r):
    return np.exp(r) / np.exp(r).sum()


class PredictiveCodingClassifier:
    def __init__(self, parameters):

        self.p = parameters

        # possible choices for transformations and priors
        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # all the representations (including the image r[0] which is not trained)
        self.r = {}

        # synaptic weights controlling reconstruction in the network
        self.U = {}

        # cost/loss function output
        self.E = 0
        # average cost per epoch during training
        self.E_avg_per_epoch = []
        # average cost per image during testing (useful?)
        self.E_avg_per_image = []

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

        print('\n' + "*** __init__ function layer setup ***")
        print("r[0] (setup) shape is " + str(np.shape(self.r[0])))
        print(" len " + str(len(self.r[0])) + '\n')

        # initialize r's and U's for hidden layers
        for i in range(1,self.n_non_input_layers):
            self.r[i] = np.random.randn(self.p.hidden_sizes[i-1],1)
            self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))

            print("r[{}] shape is ".format(i) + str(np.shape(self.r[i])))
            print(" len " + str(len(self.r[i])) + '\n')
            print("U[{}] shape is ".format(i) + str(np.shape(self.U[i]))+ '\n')

        # initialize "output" layer
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        print("o shape is " + str(np.shape(self.o)) + '\n')
        print("U_o shape is " + str(np.shape(self.U_o)) + '\n')

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

        # loop through training image dataset num_epochs times
        for epoch in range(0,self.p.num_epochs):
            # shuffle order of training set input image / output vector pairs each epoch
            N_permuted_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[N_permuted_indices]
            Y_shuffled = Y[N_permuted_indices]

            # number of training images
            num_images = X_shuffled.shape[0]

            # initialize cost function output
            self.E = 0

            print("*** train() function values and layers ***")
            print("Number of training images is {}".format(num_images) + '\n')

            # loop through training images
            for image in range(0, num_images):

                # copy first image into r[0]
                """ Fix size discrepancy: single loaded image is (784,1) and r[0] setup layer is (256,1) """

                self.r[0] = X_shuffled[image,:][:,None]
                # self.r[0] = np.random.randn(self.p.input_size,1)

                print("r[0] (loaded image) shape is " + str(np.shape(self.r[0])) + '\n')

                # initialize "output" layer o (for classification method 2 (C2))
                self.o = np.random.randn(self.p.output_size,1)
                # and final set of weights U_o to the output (C2)
                self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])
                # designate label vector
                label = Y_shuffled[image,:]

                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    # model state per layer
                    self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)

                    print("r{} reinitialized shape is ".format(layer) + str(np.shape(self.r[layer])) + '\n')

                    # loop through intermediate layers
                    # r,U updates written symmetrically for all layers including output
                    for i in range(1,n):
                        # r[i] update
                        self.r[i] = self.r[i] + (self.p.k_r / self.p.sigma[i-1] ** 2) \
                        * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                        + (self.p.k_r / self.p.sigma[i-1] ** 2) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                        - (self.p.k_r / 2) * self.g(self.r[i],self.p.alpha)[1]

                        print("After r[{}] update:".format(i))
                        print("r{} shape is ".format(i) + str(np.shape(self.r[i])) + '\n')
                        print("r{} shape is ".format(i+1) + str(np.shape(self.r[i+1])) + '\n')
                        print("U{} shape is ".format(i) + str(np.shape(self.U[i])) + '\n')
                        print("U{} shape is ".format(i+1) + str(np.shape(self.U[i+1])) + '\n')

                        # U[i] update
                        self.U[i] = self.U[i] + (self.p.k_U / self.p.sigma[i-1] ** 2) \
                        * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                        - (self.p.k_U / 2) * self.h(self.U[i],self.p.lam)[1]

                        print("After U[{}] update:".format(i))
                        print("r{} shape is ".format(i) + str(np.shape(self.r[i])) + '\n')
                        print("r{} shape is ".format(i+1) + str(np.shape(self.r[i+1])) + '\n')
                        print("U{} shape is ".format(i) + str(np.shape(self.U[i])) + '\n')
                        print("U{} shape is ".format(i+1) + str(np.shape(self.U[i+1])) + '\n')

                        # optimization function E
                        self.E = self.E + (1 / self.p.sigma[i-1] ** 2) \
                        * (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]).T.dot(self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                        + self.h(self.U[i],self.p.lam)[0] + self.g(np.squeeze(self.r[i]),self.p.alpha)[0]

                        print("E update value:" + str(self.E))
                        print("E update, sum of prior terms h(U) + g(r) shape:")

                        print((self.h(self.U[i],self.p.lam)[0] + self.g(np.squeeze(self.r[i]),self.p.alpha)[0]).shape)

                    # r[n] update (C1)
                    self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma[i-1] ** 2) \
                    * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                    - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha)[1] \
                    # classification term
                    + (self.p.k_o / 2) * (label - softmax(self.r[n]))

                    print("After r[{}] update:".format(n))
                    print("r{} shape is ".format(n) + str(np.shape(self.r[n])) + '\n')

                    # # r(n) update (C2)
                    # self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma[i-1] ** 2) \
                    # * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                    # - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha)[1] \
                    # # classification term
                    # + (self.p.k_o / 2) * (self.U_o.T.dot(label) - self.U_o.T.dot(softmax(self.U_o.dot(self.r[n]))))

                    # U[n] update (C1, C2) (identical to U[i], except index numbers)
                    self.U[n] = self.U[n] + (self.p.k_U / self.p.sigma[i-1] ** 2) \
                    * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                    - (self.p.k_U / 2) * self.h(self.U[n],self.p.lam)[1]

                    print("After U[{}] update:".format(n))
                    print("U{} shape is ".format(n) + str(np.shape(self.U[n])) + '\n')

                    # # U_o update (C2)
                    # self.U_o = self.U_o + (self.p.k_o / 2) * (label.dot(self.r[n].T) - (self.p.output_size / np.exp(self.U_o.dot(self.r[n])).sum())\
                    # * self.o.dot(self.r[n].T)) - (self.p.k_o / 2) * self.h(self.U_o,self.p.lam)[1]

                    # E update (C1)
                    for L in range(0,self.p.output_size):
                        C1 = -(np.log(softmax(self.r[n]))).dot(label[L])
                        self.E = self.E + C1

                        print('\n'+'label element {} size is: '.format(L) + str(label[L])+'\n')

                    print("Error after image {} is:".format(image) + '\n')
                    print(self.E)

                    # # E update (C2)
                    # for L in range(0,self.p.output_size):
                    #     C2 = -(np.log(softmax(self.U_o.dot(self.r[n])))).dot(label[L]) + self.h(self.U_o,self.p.lam)[0]
                    #     self.E = self.E + C2

            # adjust learning rates for r, U, or o every epoch
            # self.p.k_r += 0.05
            # self.p.k_U += 0.05
            # self.p.k_o += 0.05

            # store average cost per epoch
            self.E_avg_per_epoch.append(self.E / num_images)

        print('\n' + "Average error per each (of {}) epochs:".format(self.p.num_epochs))
        print(self.E_avg_per_epoch)
        print("Model trained.")

        return


    def test(self,X):
        '''
        Given one or more inputs, produce one or more outputs.
        '''

        # number of hidden layers
        n = self.n_hidden_layers

        # number of testing images
        num_images = X.shape[0]

        # initialize cost function output
        self.E = 0

        # initialize output dictionary
        output_r = {}

        # loop through testing images
        for image in range(0, num_images):
            # copy first image into r[0]
            self.r[0] = X[image,:]

            # initialize new r's
            for layer in range(1,self.n_non_input_layers):
                self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)

                # loop through intermediate layers
                # r updates written symmetrically for all layers including output
                for i in range(1,n):
                    # r[i] update
                    self.r[i] = self.r[i] + (self.p.k_r / self.p.sigma[i-1] ** 2) \
                    * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                    + (self.p.k_r / self.p.sigma[i-1] ** 2) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                    - (self.p.k_r / 2) * self.g(self.r[i],self.p.alpha)[1]

                    # optimization function E
                    self.E = self.E + (1 / self.p.sigma[i-1] ** 2) \
                    * (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]).T.dot(self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                    + self.h(self.U[i],self.p.lam)[0] + self.g(np.squeeze(self.r[i]),self.p.alpha)[0]

                # r(n) update (C1, C2)
                self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma[i-1] ** 2) \
                * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha)[1]

                output_r[image] = self.r[n]

            self.E_avg_per_image.append(self.E / n)

        return output_r
