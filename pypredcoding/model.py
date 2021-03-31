# Implementation of r&b 1999 predictive coding model with MNIST data.

import numpy as np
import parameters as parameters
import data as data
from matplotlib import pyplot as plt

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
    F = np.diag(1 - f.flatten()**2)
    return (f, F)


# r, U prior functions
def gauss_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Gaussian prior. """
    g_or_h = alph_or_lam * np.square(r_or_U).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U
    return (g_or_h, gprime_or_hprime)


def kurt_prior(r_or_U, alph_or_lam):
    """ Takes an argument pair of either r & alpha, or U & lambda, and returns
    a tuple of (g(r), g'(r)), or (h(U), h'(U)), respectively. Sparse kurtotic prior. """
    g_or_h = alph_or_lam * np.log(1 + np.square(r_or_U)).sum()
    gprime_or_hprime = 2 * alph_or_lam * r_or_U / (1 + np.square(r_or_U))
    return (g_or_h, gprime_or_hprime)


# softmax function
def softmax(r):
    return np.exp(r) / np.exp(r).sum()


class PredictiveCodingClassifier:
    def __init__(self, parameters):

        self.p = parameters

        # possible choices for transformations, priors
        self.unit_act = {'linear':linear_trans,'tanh':tanh_trans}
        self.prior_dict = {'gaussian':gauss_prior, 'kurtotic':kurt_prior}

        # NOTE: may use this functionality later
        # possible classification methods
        # self.class_cost_dict = {'C1':class_c1, 'C2':class_c2}

        # all the representations (including the image r[0] which is not trained)
        self.r = {}

        # synaptic weights controlling reconstruction in the network
        self.U = {}

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

        # classification method
        # self.class_cost = self.class_cost_dict[self.p.classification]
        # # if C1, how to call C1: C = C - self.class_cost(self.r[n], label)
        # # if C2, how to call C2: C = self.class_cost(self.r[n], self.U_o, label) + \
        # # (self.h(self.U_o,self.p.lam[n-1])[0])[0,0]

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

        print('\n')
        print("*** Predictive Coding Classifier ***")
        print('\n')
        print("*** Layer shapes ***")
        print('\n')
        print("r[0] shape is " + str(np.shape(self.r[0])))
        print('\n')

        # dict to contain number of neural network parameters per layer comprising the model
        self.nn_parameters_dict = {}

        # initialize r's and U's for hidden layers
        # calculate number of network parameters per layer
        for i in range(1,self.n_non_input_layers):
            self.r[i] = np.random.randn(self.p.hidden_sizes[i-1],1)
            self.U[i] = np.random.randn(len(self.r[i-1]),len(self.r[i]))
            ri = "r_{}".format(i)
            Ui = "U_{}".format(i)
            self.nn_parameters_dict[ri] = len(self.r[i])
            self.nn_parameters_dict[Ui] = np.shape(self.U[i])[0]*np.shape(self.U[i])[1]

            print("r[{}] shape is ".format(i) + str(np.shape(self.r[i])))
            print('\n')
            print("U[{}] shape is ".format(i) + str(np.shape(self.U[i])))
            print('\n')

        print("len(self.r): " + str(len(self.r)))
        print('\n')

        # initialize "output" layer
        self.o = np.random.randn(self.p.output_size,1)
        # and final set of weights to the output
        self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])

        print("o shape is " + str(np.shape(self.o)))
        print('\n')
        print("U_o shape is " + str(np.shape(self.U_o)))
        print('\n')

        # calculate total number of network parameters comprising the model (hidden layers only)
        self.n_model_parameters = sum(self.nn_parameters_dict.values())

        print("*** Network Parameters ***")
        print('\n')
        print("number of neural network parameters per hidden layer: " + str(self.nn_parameters_dict))
        print('\n')
        print("total number of hidden layer parameters: {}".format(self.n_model_parameters))
        print('\n')

        return


    def rep_cost(self):
        '''
        Uses current r/U states to compute the least squares portion of the error
        (concerned with accurate reconstruction of the input).
        '''
        E = 0
        # LSQ cost
        for i in range(0,len(self.r)-1):
            v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
            E = E + ((1 / self.p.sigma_sq[i+1]) * v.T.dot(v))[0,0]
        # priors on r[1],...,r[n]; U[1],...,U[n]
        for i in range(1,len(self.r)):
            E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])
        return E


    def class_cost_1(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C1. """
        n = self.n_hidden_layers
        C1 = -1*label[:,None].dot(np.log(softmax(self.r[n].T)))[0,0]
        return C1


    def class_cost_2(self,label):
        """ Calculates the classification portion of the cost function output of a training
        image using classification method C2, uninclusive of the prior term. """
        n = self.n_hidden_layers
        C2 = -1*label[:,None].dot(np.log(softmax((self.U_o.dot(self.r[n])).T)))[0,0]
        return C2


    def train(self,X,Y):
        '''
        X: matrix of input patterns (N_patterns x input_size)
        Y: matrix of output/target patterns (N_patterns x output_size)

        I'm pretty sure the R&B model basically requires a batch size of 1, since
        we have to fit the r's to each individual image and they are ephemeral.
        '''

        # average cost per epoch during training; just representation terms
        self.E_avg_per_epoch = []
        # average cost per epoch during training; just classification term
        self.C_avg_per_epoch = []

        # number of hidden layers
        n = self.n_hidden_layers

        print("*** Training ***")
        print('\n')

        # loop through training image dataset num_epochs times
        for epoch in range(0,self.p.num_epochs):
            # shuffle order of training set input image / output vector pairs each epoch
            N_permuted_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[N_permuted_indices]
            Y_shuffled = Y[N_permuted_indices]

            # print("y_shuffled shape is: " + '\n' + str(Y_shuffled.shape))

            # number of training images
            num_images = X_shuffled.shape[0]

            # we compute average cost per epoch (batch size = 1); separate classification
            #   and representation costs so we can compare OOM sizes
            E = 0
            C = 0

            # print("*** train() function values and layers ***")
            # print("Number of training images is {}".format(num_images) + '\n')

            print("epoch {}".format(epoch+1))
            print('\n')

            # loop through training images
            for image in range(0, num_images):

                print("image {}".format(image+1))
                print('\n')

                # copy first image into r[0]
                self.r[0] = X_shuffled[image,:][:,None]

                # print("r[0] (loaded image) shape is " + str(np.shape(self.r[0])) + '\n')

                # initialize new r's
                for layer in range(1,self.n_non_input_layers):
                    # self state per layer
                    self.r[layer] = np.random.randn(self.p.hidden_sizes[layer-1],1)

                    # print("r{} reinitialized shape is ".format(layer) + str(np.shape(self.r[layer])) + '\n')

                # initialize "output" layer o (for classification method 2 (C2))
                self.o = np.random.randn(self.p.output_size,1)
                # and final set of weights U_o to the output (C2)
                self.U_o = np.random.randn(self.p.output_size,self.p.hidden_sizes[-1])
                # designate label vector
                label = Y_shuffled[image,:]

                # print("label shape is: " + '\n' + str(label.shape))

                # loop through intermediate layers (will fail if number of hidden layers is 1)
                # r,U updates written symmetrically for all layers including output
                for i in range(1,n):
                    # r[i] update
                    self.r[i] = self.r[i] + (self.p.k_r / self.p.sigma_sq[i]) \
                    * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                    + (self.p.k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                    - (self.p.k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

                    # print("r{} update term (image{} epoch{})".format(i, image+1, epoch+1))

                    # print('\n')

                    # U[i] update
                    self.U[i] = self.U[i] + (self.p.k_U / self.p.sigma_sq[i]) \
                    * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
                    - (self.p.k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]

                    # print("U{} update term (image{} epoch{})".format(i, image+1, epoch+1))

                    # print('\n')

                # XXX DEBUG
                #print(self.r[n].shape)
                #print(self.U[n].T.shape)
                #print(self.f(self.U[n].dot(self.r[n]))[1].shape)
                #print((self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0]).shape)
                #print(self.g(self.r[n],self.p.alpha[n])[1].shape)

                # r[n] update (C1)
                self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma_sq[n]) \
                * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \
                # classification term
                # + (self.p.k_o / 2) * (label - softmax(self.r[n]))

                # print("r{} update term (image{} epoch{})".format(n, image+1, epoch+1))

                # print('\n')

                # # r(n) update (C2)
                # self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma_sq[n]) \
                # * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
                # - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \
                # # classification term
                # + (self.p.k_o / 2) * (self.U_o.T.dot(label) - self.U_o.T.dot(softmax(self.U_o.dot(self.r[n]))))

                # U[n] update (C1, C2) (identical to U[i], except index numbers)
                self.U[n] = self.U[n] + (self.p.k_U / self.p.sigma_sq[n]) \
                * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
                - (self.p.k_U / 2) * self.h(self.U[n],self.p.lam[n])[1]

                # print("U{} update term (image{} epoch{})".format(n, image+1, epoch+1))

                # print('\n')

                # print("After U[{}] update:".format(n))
                # print("U{} shape is ".format(n) + str(np.shape(self.U[n])) + '\n')

                # # U_o update (C2)
                # self.U_o = self.U_o + (self.p.k_o / 2) * (label.dot(self.r[n].T) - (self.p.output_size / np.exp(self.U_o.dot(self.r[n])).sum())\
                # * self.o.dot(self.r[n].T)) - (self.p.k_o / 2) * self.h(self.U_o,self.p.lam[n])[1]

                # calculate optimization function E

                E = self.rep_cost()
                # print("E update (image{} epoch{})".format( image+1, epoch+1))
                # print(E)
                # print('\n')

                # # when classifying using C1
                # C1 = self.class_cost_1(label)
                # E = E + C1
                #
                # print("C1 update")
                # print(C1)
                # print('\n')

                # # when classifying using C2
                # C2 = self.class_cost_2(label)
                # E = E + C2
                #
                # print("C2 update")
                # print(C2 )
                # print('\n')
                #
                # print("E total, after classifier C (image{} epoch{})".format( image+1, epoch+1))
                # print(E)
                # print('\n')


            # adjust learning rates for r, U, or o every epoch
            # self.p.k_r += 0.05
            # self.p.k_U += 0.05
            # self.p.k_o += 0.05

            # store average cost per epoch

            E_avg_per_epoch = E/num_images
            C_avg_per_epoch = C/num_images

            self.E_avg_per_epoch.append(E_avg_per_epoch)
            self.C_avg_per_epoch.append(C_avg_per_epoch)

            round_first = round(self.E_avg_per_epoch[0],1)
            round_last = round(self.E_avg_per_epoch[-1],1)

            print(round_first)
            print(round_last)

            # plot results
            plt.plot(epoch+1, E_avg_per_epoch, '.k')
            plt.title("HL = {}".format(self.n_hidden_layers) + '\n' + "k_r = {}".format(self.p.k_r) \
            + '\n' + "k_U = {}".format(self.p.k_U))
            if epoch == self.p.num_epochs-1:
                plt.text(0.4*self.p.num_epochs,0.8*round_first, "E avg initial = {}".format(round_first) + '\n' \
                + "E avg final = {}".format(round_last) + '\n' \
                + "E avg descent magnitude = {}".format(round((round_first - round_last),1)) \
                + '\n' + "Descent fold decrease = {}".format(round((round_first / round_last),1)))
                plt.xlabel("epoch ({})".format(self.p.num_epochs))
                plt.ylabel("E avg")



        print("Average representation error per each epoch ({} total), in format [E_epoch1, E_epoch2...]".format(self.p.num_epochs))
        print(self.E_avg_per_epoch)
        print('\n')
        print("Average classification error per each epoch ({} total), in format [C_epoch1, C_epoch2...]".format(self.p.num_epochs))
        print(self.C_avg_per_epoch)
        print('\n')
        print("Model trained.")
        print('\n')

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
        E = 0

        # average cost per image during testing (no classification term)
        self.E_avg_per_image = []

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
                self.r[i] = self.r[i] + (self.p.k_r / self.p.sigma_sq[i]) \
                * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
                + (self.p.k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                - (self.p.k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

                # optimization function E
                E = E + (1 / self.p.sigma_sq[i+1]) \
                * (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]).T.dot(self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
                + self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0]

            # r(n) update (C1, C2)
            self.r[n] = self.r[n] + (self.p.k_r / self.p.sigma_sq[n]) \
            * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
            - (self.p.k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1]

            output_r[image] = self.r[n]

            self.E_avg_per_image.append(E/n)

        return output_r
