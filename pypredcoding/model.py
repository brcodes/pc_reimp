# Implementation of r&b predictive coding model with MNIST data.

import numpy as np
from pypredcoding.parameters import ModelParameters as MP
import unittest
import pypredcoding.data as data


# activation functions
def linear_trans(x):
    return x

def tanh_trans(x):
    return np.tanh(x)

def F(U,r):
    MP.unit_act == 'linear'
        return U.dot(r)
    else:
        # if 'tanh'
        return 1 - np.tanh(U.dot(r))^2

# g/prime (r) and h/prime (U) priors
# the argument a represents either alpha (r) or lambda (U)
def gauss_prior(a, x):
    gp = 2 * a * x
    return gp

def kurt_prior(a, x):
    kp = 2 * a * x / (1 + x^2)
    return kp

def g(r):
    if MP.r_prior == 'gaussian'
        return MP.alpha * r^2
    else:
        return MP.alpha * np.log(1 + r^2)

def h(U):
    if MP.U_prior == 'gaussian'
        return MP.lambda * U^2
    else:
        return MP.lambda * np.log(1 + U^2)

# softmax function
def softmax(r):
    return np.exp(r) / (np.exp(r) * MP.output_size)



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

        # classification cost
        self.C = self.p.classification

        return

    def train(self,X,Y):
        '''
        X: matrix of input patterns (n_patterns x input_size)
        Y: matrix of output/target patterns (n_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        # classification cost types
        def classification_1(L, l, r):
            for i in range(0,L-1):
                C1 = l[i,:].dot(np.log(softmax(r)))
                return -C1

        def classification_2(L, l, U, r):
            for i in range(0,L-1):
                C2 = l[i,:].dot(np.log(softmax(U.dot(r))))
                return -C2 + h(U)

        n = self.n_layers

        # loop through training images
        for image in range(0, X.shape[0]):
            # copy first image into r[0]
            self.r[0] = X[image,:]

            # r,U updates can be written symmetrically for all layers including output
            # how do I skip
            for i in range(1,n-1):
                # do r update
                self.r[i] += (self.p.k_r / self.p.sigma^2) \
                * self.U[i].T.dot(F(self.U[i],self.r[i]).dot(self.r.[i-1]-self.f(self.U[i].dot(self.r[i])))) \
                + (self.p.k_r / self.p.sigma^2) * (self.f(self.U[i+1].dot(self.r[i+1])) - self.r[i]) \
                - (self.p.k_r / 2) * self.gprime(self.p.alpha,self.r[i])

                # do U update
                self.U[i] += (self.p.k_U / self.p.sigma^2) \
                * (F(self.U[i],self.r[i]).dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i])))).dot(self.r[i].T) \
                - (self.p.k_U / 2) * self.hprime(self.p.lambda,self.U[i])

                # optimization function E if classification_1
                self.E = (1 / self.p.sigma^2) \
                * (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))).T.dot(self.r[i] - self.f(self.U[i+1].dot(self.r[i+1])))
                + h(self.U[i]) + g(self.r[i]) + classification_1()

            # need to add correction term to output layer that compares to output pattern
            # r[n] update, if C1
            self.r[n] += (self.p.k_r / self.p.sigma^2) \
            * self.U[n].T.dot(F(self.U[n],self.r[n]).dot(self.r.[n-1]-self.f(self.U[n].dot(self.r[n])))) \
            - (self.p.k_r / 2) * self.gprime(self.p.alpha,self.r[n]) \
            # classification term
            + (self.p.k_r / 2) * (Y[image,:] - softmax(self.r[n]))

            # # if C2
            # # U(n) update
            # self.U_o += (self.p.k_U / 2) \
            # * Y[image,:].dot(self.r[n].T) - softmax(self.U_o[n].dot(self.r[n])) * self.p.output_size
            # - (self.p.k_U / 2) * self.hprime(self.p.lambda,self.U_o[n])

            # # r(n) update if C2
            # self.r[n] += (self.p.k_r / self.p.sigma^2) \
            # * self.U[n].T.dot(F(self.U[n],self.r[n]).dot(self.r.[n-1]-self.f(self.U[n].dot(self.r[n])))) \
            # - (self.p.k_r / 2) * self.gprime(self.p.alpha,self.r[n]) \
            # # classification term
            # + (self.p.k_r / 2) * (Y[image,:].dot(self.U_o[image] - self.U_o.T.dot(self.o) - softmax(self.r[n]))

        return


    def test(self,X):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        return
