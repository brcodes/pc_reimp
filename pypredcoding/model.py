# Implementation of r&b predictive coding model toward use with MNIST data.

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

'''
NOTE ON INSTANTIATION:
As I've currently set the parameters:
    r[0] has size parameters.input_size
    r[1] has size parameters.hidden_size[0] (so you know what size U/U.T is now)
    r[2] has size parameters.hidden_size[1]
    etc.
    r[n] MUST have size equal to parameters.output_size

When we train, each r/U pair is updated according to the equations in my notes.
For layer n ONLY, it has the standard update plus another term that uses the
output class vectors (layer n+1) and is a function of sigmoid(r[n])


    ---
    Equations
    ---

    # From Rao and Ballard 1999; Kevin Brown


    transform = self.transform_dict[]


    ---
    Hierarchical Generative Model
    ---

    I = f(U.dot(r)) + n

    ---
    Optimization Function
    ---

    E1 = 1 / (np.square(self.p.sigma)) * (I - f(Ur)).T * (I - f(Ur))
    + 1 / (np.square(self.p.sigma)) * (r[i] - r[i + 1]).T * (r[i] - r[i + 1])


    ---
    Network Dynamics and Synaptic Learning
    ---

    # Gradient descent on E with respect to r, assuming Gaussian prior
    def dr_dt():
        if self.p.f_of_x == "Linear":
            dr_dt = (self.p.k * U.T / np.square(self.p.sigma))
            * np.identity(len(?)) * (I - f(U.dot(r))
            + (self.p.k * U.T / np.square(self.p.sigma))
            * (r[i + 1] - r[i]) - (self.p.k / 2) * g_prime(r[i])
            return dr_dt
        else:
            dr_dt = ?
            return dr_dt

    # Gradient descent on E with respect to U, assuming Gaussian prior
    def dU_dt():
        if self.p.f_of_x == "Linear":
            dU_dt = (self.p.k / np.square(self.p.sigma))
            * np.identity(len(?)) * (I - f(U.dot(r)) * r[i].T
            - self.p.k * self.p.lam * U[i]
            return dU_dt
        else:
            dU_dt = ?
            return dU_dt

    ---
    Model Updates
    ---

    # r updates
    r[i] = f(U[i + 1].dot(r[i + 1]))

    # U updates
    dU = (self.p.k_U / self.p.sigma) * np.outer(F3.dot(E[i - 1]), r[i]) \
        - self.p.k_U * self.p.lam * U[i]
        U_dict[i] += dU
'''


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

        # some pseudocode here
        for i in range(X.shape[0]):
            # copy first image into r[0]
            self.r[0] = X[i,:]
            # r,U updates can be written symmetrically for all layers including output
            for layer in range(1,num_layers):
                pass
                # do r update - will look something like
                #   r[i] -> r[i] + (learning_rate/sigma^2)*F[i]*(r[i-1] - f(U[i]*r[i]))*r[i].T
                # do U update - will look something like
                #   U[i] -> U[i] - (lr/sigma^2)*F[i]*(r[i-1] - f(U[i]*r[i]))*r[i].T + lr/2*h'(U[i])
            # need to add correction term to output layer that compares to output pattern (so far
            #   all we have done is make the network reconstruct all the intermediate layers properly)
            # this update looks roughly like:
            # U[i] -> U[i] + (lr/2)*(Y[i,:]- softmax(r[output]))
        '''
        '''
        for layer in range(1, self.n_layers):

            # r update
            self.r[i] += (self.p.k_r / np.square(self.p.sigma)) \
            * self.f(i) * (self.r[i-1] - f(self.U[i].dot(self.r[i])) \
            * self.r[i].T

            # U update
            self.U[i] += (self.p.k_U / np.square(self.p.sigma)) \
            * self.f(i) * (self.r[i-1] - self.f(self.U[i].dot(self.r[i])) \
            * self.r[i].T + self.k_U / 2 * self.hprime(self.U[i])

            return
        '''

    def test(self,X):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        return
