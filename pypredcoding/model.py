# Implementation of r&b predictive coding model with MNIST data.

import numpy as np
import pypredcoding.parameters
import unittest
import pypredcoding.data as data


# activation functions
def linear_trans(x):
    return x

def tanh_trans(x):
    return np.tanh(x)

# priors
def gauss_prior(a, x):
    gp = 2 * a * x
    return gp

def kurt_prior(a, x):
    kp = 2 * a * x / (1 + np.square(x))
    return kp


class PredictiveCodingClassifier:
    def __init__(self, parameters):

        self.p = parameters

        # possible choices for transformations and priors
        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # all the representations (including the image r[0] which is not trained)
        self.r = {}
        # softmax output layer
        self.o = np.zeros(self.p.output_size)

        # synaptic weights controlling reconstruction in the network
        self.U = {}

        # actual cost/loss function (if we want to keep track)
        self.E = 0

        # priors and transforms
        self.f = self.unit_act[self.p.unit_act]
        self.gprime = self.prior_dict[self.p.r_prior]
        self.hprime = self.prior_dict[self.p.U_prior]

        self.n_layers = len(self.p.hidden_sizes) + 2

        # initialize the appropriate r's and U's
        # N.B - MAY NEED TO DO THIS PROPERLY USING PRIORS AT SOME POINT
        # input
        self.r[0] = np.zeros(self.p.input_size)

        # hidden layers
        for i in range(1,len(self.p.hidden_sizes)+1):
            self.r[i] = np.random.randn(self.p.hidden_sizes[i-1])
            self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))

        # "output" layer"
        self.o = np.random.randn(self.p.output_size)
        # final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        return


    def train(self,X,Y):
        '''
        X: matrix of input patterns (n_patterns x input_size)
        Y: matrix of output/target patterns (n_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        # filling in kb pseudocode with real objects
        for i in range(0, X.shape[0]):
            # copy first image into r[0]
            self.r[0] = X[i,:]
            # r,U updates can be written symmetrically for all layers including output
            for layer in range(1,self.p.num_layers):
                # do r update
                self.r[i] += (self.p.k_r / self.p.sigma) \
                  * self.U[i].T.dot(np.diag(1-self.f(self.U[i].dot(self.r[i]))).dot(self.r[i]-self.f(self.U[i].dot(self.r[i])))) \
                  + (self.p.k_r / self.p.sigma) * -(self.r[i]+self.f(self.U[i+1].dot(self.r[i+1]))) \
                  - self.p.k_r * self.p.alpha * self.r[i]
                # do U update
                self.U[i] += (self.p.k_U / self.p.sigma) \
                  * np.outer(np.diag(1-self.f(self.U[i].dot(self.r[i]))).dot(self.r[i]-self.f(self.U[i].dot(self.r[i])))), self.r[i]) \
                  - self.p.k_U * self.p.lambda * self.U[i]

            # need to add correction term to output layer that compares to output pattern (so far
            # all we have done is make the network reconstruct all the intermediate layers properly)
            # this update looks roughly like:
                self.U[i] += (self.p.k_U/2)*(Y[i,:]- softmax(self.r[output]))

        return


    def test(self,X):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        return
