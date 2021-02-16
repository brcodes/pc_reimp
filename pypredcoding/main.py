from data import get_mnist_data,flatten_images
from parameters import ModelParameters
from model import PredictiveCodingClassifier


if __name__ == '__main__':
    # create and modify model parameters
    p = ModelParameters()
    # load data
    X_train, y_train = get_mnist_data(frac_samp=0.000166,return_test=False)
    # do any rescaling, normalization here
    # flatten
    X_flat = flatten_images(X_train)
    # instantiate model
    pcmod = PredictiveCodingClassifier(p)
    # train
    pcmod.train(X_flat,y_train)
