from data import get_mnist_data,flatten_images,standardization_filter
from parameters import ModelParameters
from model import PredictiveCodingClassifier
import numpy as np


if __name__ == '__main__':
    # create and modify model parameters
    p = ModelParameters(hidden_sizes = [32], k_r=0.00005, k_U = 0.0005, num_epochs = 100)
    # load data
    # frac_samp 0.000166 = 10 images
    X_train, y_train = get_mnist_data(frac_samp=0.0166,return_test=False)

    # number of initial training images
    num_imgs = y_train.shape[0]
    # print(num_imgs)

    # make evenly distributed dataset of digits 0 - 9, 10 digits each
    # the following logic, which proceeds down to the "rescaling, normalization" area of main.py
    #   takes however many initial training images there are (1000 here), and extracts
    #   ten of each digit, generating an evenly distributed 10 dig x 10 images (100 image) training set

    X_dict = {}
    y_dict = {}

    for i in range(0,10):
        X_dict[i] = np.zeros(shape=(1,28,28))
        y_dict[i] = np.zeros(shape=(1,10))

    # print(X_dict[0].shape)
    # print(y_dict[0].shape)

    for i in range(0,num_imgs):
        label = y_train[i,:]
        digit = np.nonzero(label)[0][0]
        X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
        y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))

    # print("dict key sizes")
    # for i in range(0,10):
    #     print(X_dict[i][1:,:,:].shape)

    # print("dict num keys")
    # print(len(X_dict))

    X_dist = np.zeros(shape=(1,28,28))
    y_dist = np.zeros(shape=(1,10))

    # i is each unique digit, 0 - 9
    # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
    # b) only take first ten filled vectors from the value per key
    for i in range(0,10):
        X_dist = np.vstack((X_dist, X_dict[i][1:11,:,:]))
        y_dist = np.vstack((y_dist, y_dict[i][1:11,:]))

    # remove first empty vector of X_dist, y_dist to generate final image and label set
    X_dist = X_dist[1:,:,:]
    y_dist = y_dist[1:,:]

    # print(X_dist.shape)
    # print(y_dist.shape)
    # for i in range(0,100):
    #     print(y_dist[i])
    #
    # for i in range(0,12):
    #     print(X_dist[i])

    # do any rescaling, normalization here
    X_stdized = standardization_filter(X_dist)
    # flatten
    X_flat = flatten_images(X_stdized)
    # instantiate model
    pcmod = PredictiveCodingClassifier(p)
    # train
    pcmod.train(X_flat,y_dist)

    print('Total number of model parameters')
    print(pcmod.n_model_parameters)
    print('\n')
