from data import get_mnist_data,flatten_images,standardization_filter
from parameters import ModelParameters
from model import PredictiveCodingClassifier
import numpy as np


if __name__ == '__main__':
    # create and modify model parameters
    p = ModelParameters(hidden_sizes=[32,32],num_epochs=50)
    # load data
    # frac_samp 0.000166 = 10 images
    X_train, y_train = get_mnist_data(frac_samp=0.000166,return_test=False)
    # do any rescaling, normalization here
    X_stdized = standardization_filter(X_train)
    # flatten
    X_flat = flatten_images(X_stdized)
    # instantiate model
    pcmod = PredictiveCodingClassifier(p)
    # train
    pcmod.train(X_flat,y_train)

    print('Total number of model parameters')
    print(pcmod.n_model_parameters)
    print('\n')
