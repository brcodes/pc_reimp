from parameters import ModelParameters
from model import PredictiveCodingClassifier
import numpy as np
import pickle
import cProfile
import pstats
import re



def main():


    # create and modify model parameters

    p = ModelParameters(unit_act='tanh',hidden_sizes = [32,32], k_r = 0.0005, k_U = 0.005, num_epochs = 20)

    # instantiate model

    pcmod = PredictiveCodingClassifier(p)

    # load preprocessed data saved by preprocessing.py
    # for a linear model training on a linear-optimized training set of 10digs x 10imgs open "linear-10x10.pydb"
    # comment out the below three lines if using tanh model

    # linear_data_in = open('linear-10x10.pydb','rb')
    # X_train, y_train = pickle.load(linear_data_in)
    # linear_data_in.close()

    # for a tanh model training on a tanh-optimized training set of 10digs x 10imgs open "tanh-10x10.pydb"
    # comment out the below three lines if using linear model

    tanh_data_in = open('tanh-10x10.pydb','rb')
    X_train, y_train = pickle.load(tanh_data_in)
    tanh_data_in.close()

    # train

    pcmod.train(X_train, y_train)

    # display total number of model parameters after training

    print('Total number of model parameters')
    print(pcmod.n_model_parameters)
    print('\n')


if __name__ == '__main__':
    # for unabridged cProfile readout in bash shell type: 'python -m cProfile main.py'

    main()

    # for truncated cProfile readout in IDE, use logic below

    # NOTE: fix this
    # pst = pstats.Stats('restats')
    # pst.strip_dirs().sort_stats(-1).print_stats()
    # cProfile.run('main()')
