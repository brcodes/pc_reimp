# Ground-up rewrite of r&b predictive coding model toward use with MNIST data.

import numpy as np
import parameters

class Model:
    def __init__(self, dataset, iteration=1):
        # long term we will probably want to use a data "generator" which pulls batches
        #    from a large set of patterns stored on disk, which helps with RAM.  I have
        #    an example of how to do this from Nick in the EARSHOT project (I can help
        #    with coding that also.) So you won't pass in "data", you'll pass in some object
        #    that the model can call to get a training batch from.
        self.dataset = dataset
        self.iteration = iteration


if __name__ == '__main__':

    """ Layer Setup """

    # keep in mind final form, eg: I, r[i], r[i+1], l
    r_dict = {}
    U_dict = {}
    E_dict = {}

    for layer in range(1, ModelParams.num_layers + 1):
        r_dict['r{}'.format(layer)] = np.zeros((1,1), dtype=int)
        U_dict['U{}'.format(layer)] = np.random.rand(1,1)
        E_dict['E{}'.format(layer)] = np.zeros((1,1), dtype=int)

    """ Equations """

    # From Rao and Ballard 1999; Kevin Brown

    ### --- Hierarchical Generative Model --- ###

    I = f(U.dot(r)) + n

    def f(x):
        if ModelParams.f_of_x == "Linear":
            return x
        else:
            x = np.tanh(x)
            return x

    ### --- Optimization Function --- ###

    E1 = 1 / (np.square(ModelParams.sigma)) * (I - f(Ur)).T * (I - f(Ur))
    + 1 / (np.square(ModelParams.sigma)) * (r[i] - r[i + 1]).T * (r[i] - r[i + 1])

    E = E1 + g(r) + h(U)

    ### --- Prior Distributions --- ###

    # g(r) is the -log of the prior probability of r

    def g(r):
        if ModelParams.prior == "Gaussian":
            g_gauss = ModelParams.alpha * np.square(r))
            return g_gauss
        else:
            g_kurt = ModelParams.alpha * np.log(1 + np.square(r))
            return g_kurt

    # g'(r)

    def g_prime(r):
        if ModelParams.prior == "Gaussian":
            g_prime_gauss = 2 * ModelParams.alpha * r
            return g_prime_gauss
        else:
            g_prime_kurt = 2 * ModelParams.alpha * r /
            (1 + np.square(r))
            return g_prime_kurt

    # h(U) is the -log of the prior probability of U

    def h(U):
        if ModelParams.prior == "Gaussian":
            h_gauss = ModelParams.lam * np.square(U)
            return h_gauss
        else:
            h_kurt = ModelParams.lam * np.log(1 + np.square(U))
            return h_kurt

    # h'(U)

    def h_prime(U):
        if ModelParams.prior == "Gaussian":
            h_prime_gauss = 2 * ModelParams.lam * U
            return h_prime_gauss
        else:
            h_prime_kurt = 2 * ModelParams.lam * U /
            (1 + np.square(U))
            return h_prime_kurt

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

    """ Model Updates """

    ### --- r --- ###

    r[i] = f(U[i + 1].dot(r[i + 1]))

    ### --- U --- ###

    dU = (ModelParams.k_U / ModelParams.sigma) * np.outer(F3.dot(E[i - 1]), r[i]) \
        - ModelParams.k_U * ModelParams.lamb * U[i]
        U_dict[i] += dU
