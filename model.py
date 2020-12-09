# Implementation of r&b predictive coding model toward use with MNIST data.

import numpy as np
import parameters
import unittest


def linear_trans(x):
    return x

def tanh_trans(x):
    return np.tanh(x)

def gauss_prior(a,x):
    gp = 2*a*x
    return gp

def kurt_prior(a,x):
    pass


class Model:
    def __init__(self, dataset, iteration=1):
        # long term we will probably want to use a data "generator" which pulls batches
        #    from a large set of patterns stored on disk, which helps with RAM.  I have
        #    an example of how to do this from Nick in the EARSHOT project (I can help
        #    with coding that also.) So you won't pass in "data", you'll pass in some object
        #    that the model can call to get a training batch from.
        self.dataset = dataset
        self.iteration = iteration
        self.transform_dict = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior}

        self.r_dict = {}
        self.U_dict = {}
        self.E_dict = {}

        self.f = self.transform_dict[p.f_of_x]
        self.gprime = self.prior_dict[p.r_prior]
        self.hprime = self.prior_dict[p.U_prior]


        """
        Equations
        """

        # From Rao and Ballard 1999; Kevin Brown

        transform = self.transform_dict[]

        ### --- Hierarchical Generative Model --- ###

        # def f(x):
        #     if ModelParams.f_of_x == "Linear":
        #         return x
        #     else:
        #         x = np.tanh(x)
        #         return x
        #
        # I = f(U.dot(r)) + n

        ### --- Optimization Function --- ###

        E1 = 1 / (np.square(ModelParams.sigma)) * (I - f(Ur)).T * (I - f(Ur))
        + 1 / (np.square(ModelParams.sigma)) * (r[i] - r[i + 1]).T * (r[i] - r[i + 1])

        E = E1 + g(r) + h(U)

        ### --- Prior Distributions --- ###

        # g(r) is the -log of the prior probability of r

        # def g(r):
        #     if ModelParams.prior == "Gaussian":
        #         g_gauss = ModelParams.alpha * np.square(r))
        #         return g_gauss
        #     else:
        #         g_kurt = ModelParams.alpha * np.log(1 + np.square(r))
        #         return g_kurt
        #
        # # g'(r)
        #
        # def g_prime(r):
        #     if ModelParams.prior == "Gaussian":
        #         g_prime_gauss = 2 * ModelParams.alpha * r
        #         return g_prime_gauss
        #     else:
        #         g_prime_kurt = 2 * ModelParams.alpha * r /
        #         (1 + np.square(r))
        #         return g_prime_kurt
        #
        # # h(U) is the -log of the prior probability of U
        #
        # def h(U):
        #     if ModelParams.prior == "Gaussian":
        #         h_gauss = ModelParams.lam * np.square(U)
        #         return h_gauss
        #     else:
        #         h_kurt = ModelParams.lam * np.log(1 + np.square(U))
        #         return h_kurt
        #
        # # h'(U)
        #
        # def h_prime(U):
        #     if ModelParams.prior == "Gaussian":
        #         h_prime_gauss = 2 * ModelParams.lam * U
        #         return h_prime_gauss
        #     else:
        #         h_prime_kurt = 2 * ModelParams.lam * U /
        #         (1 + np.square(U))
        #         return h_prime_kurt

        ### --- Network Dynamics and Synaptic Learning --- ###

        # Gradient descent on E with respect to r, assuming Gaussian prior

        def dr_dt():
            if ModelParams.f_of_x == "Linear":
                dr_dt = (ModelParams.k * U.T / np.square(ModelParams.sigma))
                * np.identity(len(?)) * (I - f(U.dot(r))
                + (ModelParams.k * U.T / np.square(ModelParams.sigma))
                * (r[i + 1] - r[i]) - (ModelParams.k / 2) * g_prime(r[i])
                return dr_dt
            else:
                dr_dt = ?
                return dr_dt

        # Gradient descent on E with respect to U, assuming Gaussian prior

        def dU_dt():
            if ModelParams.f_of_x == "Linear":
                dU_dt = (ModelParams.k / np.square(ModelParams.sigma))
                * np.identity(len(?)) * (I - f(U.dot(r)) * r[i].T
                - ModelParams.k * ModelParams.lam * U[i]
                return dU_dt
            else:
                dU_dt = ?
                return dU_dt

        """
        Model Updates
        """

        ### --- r --- ###

        r[i] = f(U[i + 1].dot(r[i + 1]))

        ### --- U --- ###

        dU = (ModelParams.k_U / ModelParams.sigma) * np.outer(F3.dot(E[i - 1]), r[i]) \
            - ModelParams.k_U * ModelParams.lamb * U[i]
            U_dict[i] += dU


    def setup(self,p):
    # to set up the model in a way other than specified by the init function


    def train(self):
        '''
        This uses gradient descent on r,U on a set of data in the form
        of input,output (both vectors).
        '''
        # for image,target in images.items():
        #   r[0] = image
        #   r[nlayers + 1] = target
        #   for i in range(1,nlayers):
        #       do r[i] update with r[i-1], r[i]
        #       do U[i] update
        # calling the prior dict for r,U:
        self.gprime(p.alpha,r)
        self.hprime(p.lam,U)
        pass



    def test(self):
        '''
        Given one or more inputs, produce one or more outputs.
        '''
        # dummy inputs and parameters
        input_images = [np.array([0,0,0,0,0,0,0,0,1]), np.array([0,0,0,1,0])]
        num_layers = 3
        layer_sizes = [32, 128, 256]
        num_classes = 10

        U_dict = {}
        r_dict = {}

        for img in range(0, len(input_images)):
            # adding the image vectors to dictionary of r's as r0's
            r_dict['r0.{}'.format(img+1)] = input_images[img]

            # take number of layers and create like-indexed r's and U's
            for layer in range(0, num_layers):
                n = layer_sizes[layer]
                # r is a row vector whose length is the number of input-estimating neurons in its layer
                r_dict['r{}.{}'.format(layer+1,img+1)] = np.zeros(n)
                # U1 matrix width is the length of the original image vector (r0)
                m = len(input_images[img])

                if layer == 0:
                    # U1 matrix dimensions
                    U_dict['U{}.{}'.format(layer+1,img+1)] = np.random.rand(m,n)
                    # Subsequent U(1+i) matrix dimensions are m, n where m is the size of r (n) in the previous layer
                else:
                    U_dict['U{}.{}'.format(layer+1,img+1)] = np.random.rand(layer_sizes[layer-1],n)

            output_vec = np.zeros(num_classes)

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
