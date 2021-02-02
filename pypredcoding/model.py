# Implementation of r&b predictive coding model with MNIST data.

import numpy as np
import pypredcoding.parameters as parameters
import unittest
import pypredcoding.data as data


# activation functions
def linear_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Linear transformation. """
    f = U_dot_r
    F = U_dot_r
    return (f, F)

def tanh_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Tanh transformation. """
    f = np.tanh(U_dot_r)
    F = 1 - np.tanh(U_dot_r) ** 2
    return (f, F)

# r, U prior functions
def gauss_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior. """
    g_or_h = alph_or_lam * r_or_U ** 2
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)

def kurt_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior. """
    g_or_h = alph_or_lam * np.log(1 + r_or_U ** 2)
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + r_or_U ** 2)
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

        # actual cost/loss function (if we want to keep track)
        self.E = 0

        # priors and transforms
        self.f = self.unit_act[self.p.unit_act]
        # how to call f(x): self.f(self.U.dot(self.r))[0]
        # how to call F(x): self.f(self.U.dot(self.r))[1]

        self.g = self.prior_dict[self.p.r_prior]
        # how to call g(r): self.g(self.r,self.p.alpha)[0]
        # how to call g'(r): self.g(self.r,self.p.alpha)[1]

        self.h = self.prior_dict[self.p.U_prior]
        # how to call h(U): self.h(self.U,self.p.lamba)[0]
        # how to call h'(U): self.h(self.U,self.p.lambda)[1]

        self.n_layers = len(self.p.hidden_sizes) + 2

        # initialize the appropriate r's and U's
        # N.B - MAY NEED TO DO THIS PROPERLY USING PRIORS AT SOME POINT
        # input - how is this done? check r&b
        self.r[0] = np.zeros(self.p.input_size)

        # hidden layers
        for i in range(1,len(self.p.hidden_sizes)+1):
            self.r[i] = np.random.randn(self.p.hidden_sizes[i-1])
            self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))

        # "output" layer
        self.o = np.random.randn(self.p.output_size)
        # final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        return

    def train(self,X,Y):
        '''
        X: matrix of input patterns (N_patterns x input_size)
        Y: matrix of output/target patterns (N_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        n = self.n_layers

        # loop through training image dataset num_epochs times
        for epoch in range(0,self.p.num_epochs):
            # shuffle all input and output pairs each epoch
            N_permuted_indices = np.random.permutation(len(X.shape[0]))
            X_shuffled = X[N_permuted_indices]
            Y_shuffled = Y[N_permuted_indices]

            # loop through training images
            for image in range(0, X.shape[0]):
                # copy first image into r[0]
                self.r[0] = X_shuffled[image,:]
                label = Y_shuffled[image,:]

                # loop through intermediate layers
                # r,U updates written symmetrically for all layers including output
                for i in range(1,n):
                    # r[i] update
                    self.r[i] += (self.p.k_r / self.p.sigma ** 2) \
                    * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r.[i-1]-self.f(self.U[i].dot(self.r[i]))[0])) \
                    + (self.p.k_r / self.p.sigma ** 2) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                    - (self.p.k_r / 2) * self.g(self.r[i],self.p.alpha)[1]

                    # U[i] update
                    self.U[i] += (self.p.k_U / self.p.sigma ** 2) \
                    * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                    - (self.p.k_U / 2) * self.h(self.U[i],self.p.lambda)[1]

                    # optimization function E
                    self.E += (1 / self.p.sigma ** 2) \
                    * (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]).T.dot(self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                    + self.h(self.U[i],self.p.lambda)[0] + self.g(self.r[i],self.p.alpha)[0]

                # # r[n] update (C1)
                # self.r[n] += (self.p.k_r / self.p.sigma ** 2) \
                # * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1]-self.f(self.U[n].dot(self.r[n]))[0])) \
                # - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha)[1] \
                # # classification term
                # + (self.p.k_o / 2) * (label - softmax(self.r[n]))

                # r(n) update (C2)
                self.r[n] += (self.p.k_r / self.p.sigma ** 2) \
                * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1]-self.f(self.U[n].dot(self.r[n]))[0])) \
                - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha)[1] \
                # classification term
                for L in range(0,self.p.output_size):
                    rC2 += label[L].dot((self.U_o[L]-self.U_o.T.dot(self.o)) / (np.exp(self.U_o.dot(self.r[n])).sum()))
                + (self.p.k_o / 2) * rC2

                # U[n] update (C1,C2) (identical to U[i], except index numbers)
                self.U[n] += (self.p.k_U / self.p.sigma ** 2) \
                * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                - (self.p.k_U / 2) * self.h(self.U[n],self.p.lambda)[1]

                # U(o) update (C2)
                self.U_o += (self.p.k_o / 2) * (label.dot(self.r[n].T) - (self.p.output_size / np.exp(self.U_o.dot(self.r[n])).sum())) \
                * self.o.dot(self.r[n].T)) - (self.p.k_o / 2) * self.h(self.U_o,self.p.lambda)[1]

                # # E update (C1)
                # for L in range(0,self.p.output_size):
                #     C1 = -label[L].dot(np.log(softmax(r[n])))
                #     self.E += C1

                # E update (C2)
                for L in range(0,self.p.output_size):
                    C2 = -label[L].dot(np.log(softmax(self.U_o.dot(r[n])))) + self.h(self.U_o,self.p.lambda)[0]
                    self.E += C2

        return


    def test(self,X):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        n = self.n_layers



        return
