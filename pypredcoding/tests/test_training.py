"""
Created on Thu Feb  4 15:49:21 2021

@author: everett
"""

""" pypredcoding model v1 test utility """

import unittest
import pypredcoding.data as data
import pypredcoding.parameters as parameters
import pypredcoding.model as model


class TestTraining(unittest.TestCase):

    def setUp(self):
        # use the default parameters
        self.p = parameters.ModelParameters()
        # create the model
        self.model = model.PredictiveCodingClassifier(self.p)

    def test_training(self):

        # load MNIST data (0.000166 * 60,000 = ~10 images)
        X_train, y_train = data.get_mnist_data(frac_samp=0.000166,return_test=False)

        # flatten X_train to N_patterns x 784
        X_flat = data.flatten_images(X_train)

        # train model
        self.model.train(X_flat, y_train)
