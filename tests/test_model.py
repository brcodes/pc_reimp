import unittest
import numpy as np
from pc_reimp.model import Model
from pc_reimp.parameters import ModelParams

'''
N.B.: This can't be run until the parameters and Model constructor have
stabilized.
'''

class TestModel(unittest.testcase):

    def setUp(self):
        # use the default parameters
        self.p = ModelParams()
        # create the model
        self.model = Model(self.p)

    def test_sizes(self):
        # create dummy input/output to check sizing
        input = np.random.randn(self.p.input_size)
        output = np.random.randn(self.p.output_size)
        # check multiplications in the instantiated model
        # layers are 0,1,...,n,n+1
        #   - layer 0 is the input (not trained)
        #   - layer n is the output layer
        #   - layer n+1 is the class labels/output pattern, which is the
        #       same size as layer n so we don't need to
        n_layers = self.model.num
        # shove the fake input vector into the correct slot
        self.model.r[0] = input
        for i in range(0,len(self.model.hidden_sizes)+1):
            rhat = np.dot(self.model.U[i+1],self.model.r[i+1])
            self.assertEqual(rhat,self.model.r[i])
