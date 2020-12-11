import unittest
import numpy as np
from pypredcoding.model import PredictiveCodingNetwork
from pypredcoding.parameters import ModelParameters


class TestModel(unittest.TestCase):

    def setUp(self):
        # use the default parameters
        self.p = ModelParameters()
        # create the model
        self.model = PredictiveCodingNetwork(self.p)

    def test_sizes(self):
        # create dummy input
        input = np.random.randn(self.p.input_size)
        # check multiplications in the instantiated model
        # input <-> first hidden
        rhat = np.dot(self.model.U[1],self.model.r[1])
        self.assertEqual(len(rhat),len(input))
        # remaining layers
        for i in range(2,len(self.model.r)):
            rhat = np.dot(self.model.U[i],self.model.r[i])
            self.assertEqual(len(rhat),len(self.model.r[i-1]))
