# Ground-up rewrite of r&b predictive coding model toward use with MNIST data.

import numpy as np

class Model:
    def __init__(self, data, inpveclen=1, numclasses=10, numlayers=2, iteration=1):
        self.data = data
        self.iteration = iteration
        self.inpveclen = inpveclen
        self.numclasses = numclasses
        self.numlayers = numlayers
        for i in range(1, (numlayers + 1)):
            self.layersize = 2 ** (2 * i + 3)
        self.sigma = np.std((I - f(Ur)).T * (I - f(Ur)))
        self.alpha = " related to variance of Gaussian priors "
        self.lambda = " related to variance of Gaussian priors "
        self.U = np.random.rand()
        self.r = np.zeros()
        self.k =




    """ Inputs to the Model """

    # ---Size of input image vector--- #
    # Number of rows (m) are 3rd class argument; default at 1
    inpvecwidth = 1 # Number of columns (n) will default at 1

    # ---Number of classes--- #
    # MNIST image data consist of digits 0-9
    # There are 10 classes, one for each possible digit.

    # ---Number of r, U layers--- #
    # Number of predictive estimator modules
    # We'll start with 2

    # ---Sizes of r1,...,rN--- #

    """ Equations """

    # From Rao and Ballard 1999; Kevin Brown

    # Eq (1)

    I = f(Ur) + n

    # Optimization function

    E1 = 1/(np.square(self.sigma)) * (I - f(Ur)).T * (I - f(Ur))

    E = E1 + g(r) + h(U)

    # Prior distributions

    def g(r):
        for i in range(0, numlayers + 1):
        g_output = self.alpha * np.log(1 + np.square(r))
        return g_output

    """ Model """

    # Create empty input vector

    model_input_vector = np.zeros(shape=(inpveclen,inpvecwidth))
