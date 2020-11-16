# Ground-up rewrite of r&b predictive coding model toward use with MNIST data.

import numpy as np

class Model:
    def __init__(self, dataset, iteration=1):
        # long term we will probably want to use a data "generator" which pulls batches
        #    from a large set of patterns stored on disk, which helps with RAM.  I have
        #    an example of how to do this from Nick in the EARSHOT project (I can help
        #    with coding that also.) So you won't pass in "data", you'll pass in some object
        #    that the model can call to get a training batch from.
        self.dataset = dataset
        self.iteration = iteration
        # if we are passing in the data - or a location to it (like a generator that will pull samples from files),
        #   you don't need to tell the model the inputsize - it can sniff it from one of the patterns
        # ditto with numclasses - just look at the length of any of the output vectors
        self.U = np.random.rand()
        self.r = np.zeros()



# this is a common trick that lets you combine "main loop stuff" with objects you
#   can import later.
if __name__ == '__main__':

    """ Inputs to the Model """

    # ---Size of input image vector--- #
    # Number of rows (m) are 3rd class argument; default at 1
    #inpvecwidth = 1 # Number of columns (n) will default at 1

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
    # N.B. - we also need the derivative of any of the priors with respect to their
    #   arguments

    def g(r):
        for i in range(0, numlayers + 1):
        g_output = self.alpha * np.log(1 + np.square(r))
        return g_output

    """ Model """

    # Create empty input vector

    model_input_vector = np.zeros(shape=(inpveclen,inpvecwidth))
