from parameters import ModelParameters
from model import PredictiveCodingClassifier
import numpy as np
import pickle
import cProfile
import pstats
import re



def main():

    """
    Set Model Parameters
    """

    # create and modify model parameters

    # #constant learning rates, optimal for linear model without classification
    # #r 0.0005, U 0.005, o 0.0005; (these values are ModelParameters() default values)
    # p = ModelParameters(unit_act='linear',hidden_sizes = [32,10], num_epochs = 100)

    #constant learning rates, optimal for tanh model without classification
    #r 0.05, U 0.05 o 0.05
    p = ModelParameters(unit_act='tanh',
        hidden_sizes = [32,10], num_epochs = 50,
        k_r_sched = {'constant':{'initial':0.05}},
        k_U_sched = {'constant':{'initial':0.05}},
        k_o_sched = {'constant':{'initial':0.0005}})

    # #kurtotic priors
    # #r 0.05, U 0.05 o 0.05
    # p = ModelParameters(unit_act='tanh',r_prior = 'kurtotic', U_prior = 'kurtotic',
    #     hidden_sizes = [32,10], num_epochs = 1000,
    #     k_r_sched = {'constant':{'initial':0.05}},
    #     k_U_sched = {'constant':{'initial':0.05}},
    #     k_o_sched = {'constant':{'initial':0.0005}})

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

    """
    Pickle In
    """

    # load preprocessed data saved by preprocessing.py
    # for a linear model training on a linear-optimized training set of 10digs x 10imgs open "linear_10x10.pydb"
    # comment out the below three lines if using tanh model

    # linear_data_in = open('linear_10x10.pydb','rb')
    # X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(linear_data_in)
    # linear_data_in.close()


    # for a tanh model training on a tanh-optimized training set of 10digs x 10imgs open "tanh_10x10.pydb"
    # comment out the below three lines if using linear model

    tanh_data_in = open('tanh_100x10.pydb','rb')
    X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
    tanh_data_in.close()

    """
    Train
    """

    # instantiate model
    pcmod = PredictiveCodingClassifier(p)

    # train on training set
    pcmod.train(X_train, y_train)


    """
    Pickle Out
    """

    # pickle output model
    # MUST uncomment desired names of parameters in the model

    #model size
    model_size = '[32.10]'

    #transformation function
    transform_type = 'tanh'
    # transform_type = 'linear'

    #prior type
    prior_type = 'gauss'
    # prior_type = 'kurt'

    #classification method
    # class_type = 'NC'
    # class_type = 'C1'
    class_type = 'C2'

    #trained or untrained
    trained = 'T'
    # trained = 'nt'

    #number of epochs if trained (if not, use -)
    # num_epochs = '1000e'
    # num_epochs = '100e'
    num_epochs = '50e'
    # num_epochs = '-'

    #dataset trained on if trained (if not, use -)
    training_dataset = 'tanh100x10'
    # training_dataset = 'tanh10x10'
    # training_dataset = '-'
    
    #evaluated or not evaluated with evaluate() (should occur in evaluation.py, so likely choose ne here in main.py)
    # evaluated = 'E'
    evaluated = 'ne'

    #images evaluated against, if evaluated (if not, use -)
    # eval_dataset = 'tanh100x10'
    # eval_dataset = 'tanh10x10'
    eval_dataset = '-'

    #used or not used for prediction with predict() (should occur in prediction.py, so likely choose np here in main.py)
    # used_for_pred = 'P'
    used_for_pred = 'np'

    #images predicted, if used for prediction (if not, use -)
    #images 1-5 from April/May exps
    # pred_dataset = '5imgs'
    pred_dataset = '-'

    #extra identifier for any particular or unique qualities of the model object
    # extra_tag = 'randUo'
    extra_tag = 'pipeline_test'
    # extra_tag = '-'


    pcmod_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
        trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
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
