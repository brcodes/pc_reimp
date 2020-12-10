# Implementation of r&b predictive coding model toward use with MNIST data.

import numpy as np
import parameters
import unittest


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


class Model:
    def __init__(self, parameters):

        self.p = parameters

        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # I've dropped the "dict" so we don't have to type so much
        self.r = {}
        self.U = {}
        self.E = 0

        self.f = self.transform_dict[self.p.unit_act]
        self.gprime = self.prior_dict[self.p.r_prior]
        self.hprime = self.prior_dict[self.p.U_prior]

        self.num_layers = len(self.p.hidden_sizes) + 1

        # set variables
        n_layers = self.num_layers
        input_size = self.p.input_size
        hidden_sizes = self.p.hidden_sizes
        output_size = self.p.output_size

        # create empty r and U vectors of the correct size and index by layer
        for i in range(0, n_layers-1):

            # create r0 (r[0]) from input image size
            if i == 0:
                self.r[0] = np.random.randn(input_size)

            # r[1] through r[n-1] are row vectors whose lengths are the number of neurons in their layer
            else:
                self.r[i] = np.random.randn(hidden_sizes[i-1])

            self.r[n_layers] = np.random.randn(output_size)

            # U[1] matrix width and length variables
            m = input_size
            n = hidden_sizes[i]

            # create U[1] matrix
            if i == 0:
                self.U[1] = np.random.randn(m,n)

            # Subsequent U[i+1] matrix dimensions are m, n, where m is the size of n (r) in the previous layer
            else:
                # if final layer, n is output size
                if i == n_layers-1:
                    self.U[n_layers] = np.random.randn(hidden_sizes[i-1], output_size)

                else:
                    self.U[i+1] = np.random.randn(hidden_sizes[i-1],n)

            """
            This loop did not create matrix objects for classification layer n + 1,
            should they be created here?
            """

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
        '''

        """
        Equations
        """
        # From Rao and Ballard 1999; Kevin Brown


        transform = self.transform_dict[]


        ### --- Hierarchical Generative Model --- ###
        I = f(U.dot(r)) + n


        ### --- Optimization Function --- ###
        E1 = 1 / (np.square(self.p.sigma)) * (I - f(Ur)).T * (I - f(Ur))
        + 1 / (np.square(self.p.sigma)) * (r[i] - r[i + 1]).T * (r[i] - r[i + 1])


        ### --- Network Dynamics and Synaptic Learning --- ###
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

        """
        Model Updates
        """

        # r updates
        r[i] = f(U[i + 1].dot(r[i + 1]))

        # U updates
        dU = (self.p.k_U / self.p.sigma) * np.outer(F3.dot(E[i - 1]), r[i]) \
            - self.p.k_U * self.p.lam * U[i]
            U_dict[i] += dU


    def setup(self,self.p):
    # to set up the model in a way other than specified by the init function


    def train(self):
        '''
        This uses gradient descent on r,U on a set of data in the form
        of input,output (both vectors).
        '''
        # we will use a data "generator" which pulls batches from a large set of patterns stored on disk

        # for image,target in images.items():
        #   r[0] = image
        #   r[nlayers + 1] = target
        #   for i in range(1,nlayers):
        #       do r[i] update with r[i-1], r[i]
        #       do U[i] update
        # calling the prior dict for r,U:
        self.gprime(self.p.alpha,r)
        self.hprime(self.p.lam,U)
        pass



    def test(self):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        # we will use a data "generator" which pulls batches from a large set of patterns stored on disk

        pass



if __name__ == '__main__':

# eg
# if __name__ == '__main__':
#   data = data.GetData()
#   p = ModelParameters(...)
#   m = Model(data,p)
#   m.train()
#   look at results of model
#   m.test()
