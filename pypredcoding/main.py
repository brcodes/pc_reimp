from parameters import ModelParameters
from model import PredictiveCodingClassifier
import numpy as np
import pickle
import cProfile
import pstats
import re



def main():


    # create and modify model parameters

    #constant learning rates, optimal for linear model without classification
    #r 0.0005, U 0.005, o 0.0005; (these values are ModelParameters() default values)
    p = ModelParameters(unit_act='linear',hidden_sizes = [32,10], num_epochs = 100)

    # #constant learning rates, optimal for tanh model without classification
    # #r 0.05, U 0.05 o 0.05
    # p = ModelParameters(unit_act='tanh',hidden_sizes = [32,32], num_epochs = 30,
    #     k_r_sched = {'constant':{'initial':0.05}},
    #     k_U_sched = {'constant':{'initial':0.05}},
    #     k_o_sched = {'constant':{'initial':0.05}})

    # #step decay learning rates for tanh model (has not been optimized)
    # p = ModelParameters(unit_act='tanh',hidden_sizes = [32,32], num_epochs = 400,
    #     k_r_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}},
    #     k_U_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}},
    #     k_o_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}})

    # #step decay learning rates for linear model (has not been optimized)
    # p = ModelParameters(unit_act='linear',hidden_sizes = [32,32], num_epochs = 400,
    #     k_r_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}},
    #     k_U_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}},
    #     k_o_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}})

    # #polynomial decay learning rates for tanh model (has not been optimized)
    # p = ModelParameters(unit_act='tanh',hidden_sizes = [32,32], num_epochs = 400,
    #     k_r_sched = {'poly':{'initial':0.05,'max_epochs':400,'poly_power':1}},
    #     k_U_sched = {'poly':{'initial':0.05,'max_epochs':400,'poly_power':1}},
    #     k_o_sched = {'poly':{'initial':0.05,'max_epochs':400,'poly_power':1}})

    # #polynomial decay learning rates for linear model (has not been optimized)
    # p = ModelParameters(unit_act='linear',hidden_sizes = [32,32], num_epochs = 400,
    #     k_r_sched = {'poly':{'initial':0.05,'max_epochs':100,'poly_power':1}},
    #     k_U_sched = {'poly':{'initial':0.05,'max_epochs':100,'poly_power':1}},
    #     k_o_sched = {'poly':{'initial':0.05,'max_epochs':100,'poly_power':1}})


    # instantiate model

    pcmod = PredictiveCodingClassifier(p)


    # load preprocessed data saved by preprocessing.py
    # for a linear model training on a linear-optimized training set of 10digs x 10imgs open "linear_10x10.pydb"
    # comment out the below three lines if using tanh model

    linear_data_in = open('linear_10x10.pydb','rb')
    X_train, y_train = pickle.load(linear_data_in)
    linear_data_in.close()

    # for a tanh model training on a tanh-optimized training set of 10digs x 10imgs open "tanh_10x10.pydb"
    # comment out the below three lines if using linear model

    # tanh_data_in = open('tanh_10x10.pydb','rb')
    # X_train, y_train = pickle.load(tanh_data_in)
    # tanh_data_in.close()


    # train

    pcmod.train(X_train, y_train)


    # predict

    # test_image = X_train[0,:]
    # pcmod.predict(test_image ,test_label, 1000)


    # pickle trained model

    pcmod_out = open('pcmod_trained.pydb','wb')
    pickle.dump(pcmod, pcmod_out)
    pcmod_out.close()

    

if __name__ == '__main__':
    # for unabridged cProfile readout in bash shell type: 'python -m cProfile main.py'

    main()

    # for truncated cProfile readout in IDE, use logic below

    # NOTE: fix this
    # pst = pstats.Stats('restats')
    # pst.strip_dirs().sort_stats(-1).print_stats()
    # cProfile.run('main()')
