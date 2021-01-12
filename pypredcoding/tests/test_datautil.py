import unittest
import numpy as np
from pypredcoding.data import *


class TestDataUtil(unittest.TestCase):

    # Note: test_datautil.py will currently fail 7/7 tests using pytest if
    # in SetUp(), get_mnist_data() argument return_test = True. Anywhere self.X.shape is used,
    # the following "AttributeError: 'tuple' object has no attribute 'shape'" will occur.
    # This occurs because self.X is now the tuple (X_train[:n_train,:,:],y_train[:n_train,:])

    def setUp(self):
        # grab a few MNIST images
        self.X,self.y = get_mnist_data(frac_samp=0.1,return_test=False)
        self.n_train = self.X.shape[0]


    def test_mnist_read(self):
        '''
        Make sure that the MNIST read returns what we think it should,
        size-wise.
        '''
        self.assertEqual(self.X.shape,(self.n_train,28,28))
        self.assertEqual(self.y.shape,(self.n_train,10))


    def test_flatten(self):
        '''
        Makes sure flattening works using MNIST images.
        '''
        flat_X = flatten_images(self.X)
        self.assertEqual(flat_X.shape,(self.n_train,28*28))


    def test_inflate(self):
        '''
        Inflates the target patterns from shape (N,10) to (N,5,2) [shape_2d!=None].
        Inflates flattened sample images from shape (N,x*y) to (N,x,y), where x = y
        (N square images generated)[shape_2d=None].
        '''
        inf_y = inflate_vectors(self.y,shape_2d=(5,2))
        self.assertEqual(inf_y.shape,(self.n_train,5,2))

        flat_X = flatten_images(self.X)
        inf_X_sq = inflate_vectors(flat_X,shape_2d=None)
        self.assertEqual(inf_X_sq.shape,self.X.shape)

    def test_rescale(self):
        '''
        Makes sure rescaling generates the correct vector array shape, as well as
        min = 0 and max = 1.
        '''
        flat_X = flatten_images(self.X)
        resc_img = rescale_images(flat_X)
        self.assertEqual(resc_img.shape,(self.n_train,28*28))
        self.assertEqual(resc_img.min(),0)
        self.assertEqual(resc_img.max(),1)

    def test_DoG(self):
        '''
        Makes sure applying DoG generates the correct image array shape.
        '''
        DoG_X = apply_DoG(self.X, (5,5), sigma1=1.3, sigma2=2.6)
        self.assertEqual(DoG_X.shape,(self.n_train,28,28))

    def test_standardization(self):
        '''
        Makes sure applying image standardization generates the correct image array shape.
        '''
        Stdized_X = apply_standardization(self.X)
        self.assertEqual(Stdized_X.shape,(self.n_train,28,28))

    def test_ZCA(self):
        '''
        Makes sure applying ZCA generates the correct image array shape.
        '''
        ZCA_X = apply_ZCA(self.X)
        self.assertEqual(ZCA_X.shape,(self.n_train,28,28))
