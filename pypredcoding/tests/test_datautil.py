import unittest
import numpy as np
from pypredcoding.data import *


class TestDataUtil(unittest.TestCase):

    def setUp(self):
        # grab a few MNIST images
        self.X,self.y = get_mnist_data(frac_samp=None,return_test=True)


    def test_mnist_read(self):
        '''
        Make sure that the MNIST read returns what we think it should,
        size-wise.
        '''
        self.assertEqual(self.X.shape,(n_train,28,28))
        self.assertEqual(self.y.shape,(n_train,10))


    def test_flatten(self):
        '''
        Makes sure flattening works using MNIST images.
        '''
        flat_X = flatten_images(self.X)
        self.assertEqual(flat_X.shape,(n_train,28*28))


    def test_inflate(self):
        '''
        Inflates the target patterns from shape (N,10) to (N,5,2).
        '''
        inf_y = inflate_vectors(self.y,shape_2d=(5,2))
        self.assertEqual(inf_y.shape,(n_train,5,2))
