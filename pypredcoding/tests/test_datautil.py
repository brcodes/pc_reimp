import unittest
import numpy as np
from pypredcoding.data import *


class TestDataUtil(unittest.TestCase):

    def setUp(self):
        # grab a few MNIST images
        (self.X,self.y),(self.X_test,self.y_test) = get_mnist_data(frac_samp=0.1,return_test=True)
        self.n_train = self.X.shape[0]

    def test_mnist_read_trainmode(self):
        '''
        Make sure that the MNIST read returns what we think it should,
        size-wise.
        '''
        self.assertEqual(self.X.shape,(self.n_train,28,28))
        self.assertEqual(self.y.shape,(self.n_train,10))

    def test_mnist_read_testmode(self):
        '''
        Makes sure that if get_mnist_data() is used to load test images in addition to
        training images (i.e. if return_test=True), that size of training set > testing set.
        This ensures that get_mnist_data() loads data successfully in either "train" or "test" mode.
        '''
        self.assertGreater(self.n_train,self.X_test.shape[0])
        self.assertGreater(self.y.shape[0],self.y_test.shape[0])


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
        resc_img = rescaling_filter(flat_X)
        self.assertEqual(resc_img.shape,(self.n_train,28*28))
        self.assertEqual(resc_img.min(),0)
        self.assertEqual(resc_img.max(),1)

    def test_DoG(self):
        '''
        Makes sure applying DoG generates the correct image array shape.
        '''
        DoG_X = diff_of_gaussians_filter(self.X, (5,5), sigma1=1.3, sigma2=2.6)
        self.assertEqual(DoG_X.shape,(self.n_train,28,28))

    def test_standardization(self):
        '''
        Makes sure applying image standardization generates the correct image array shape.
        '''
        Stdized_X = standardization_filter(self.X)
        self.assertEqual(Stdized_X.shape,(self.n_train,28,28))

    def test_ZCA(self):
        '''
        Makes sure applying ZCA generates the correct image array shape.
        '''
        ZCA_X = zca_filter(self.X)
        self.assertEqual(ZCA_X.shape,(self.n_train,28,28))
