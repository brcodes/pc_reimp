from parameters import ModelParameters
from model_c1 import TiledPredictiveCodingClassifier
import numpy as np
import pickle
import cProfile
import pstats
import datetime



def main():

    """
    Set Model Parameters
    """

    # create and modify model parameters

    # TANH TILED polynomial
    p = ModelParameters(unit_act='tanh',r_prior = 'gaussian', U_prior = 'gaussian', input_size=576,
        hidden_sizes = [18432,10], classification = 'C1', num_epochs = 10, tile_offset = 6,
        k_r_sched = {'poly':{'initial':0.005,'max_epochs':40,'poly_power':1}},
        k_U_sched = {'poly':{'initial':0.005,'max_epochs':40,'poly_power':1}},
        k_o_sched = {'poly':{'initial':0.0005,'max_epochs':40,'poly_power':1}})

    # model_size = '[36.10]'
    # model_size = '[288.10]'
    # model_size = '[2304.10]'
    model_size = '[18432.10]'

    # #step decay learning rates for tanh model (has not been optimized)
    # p = ModelParameters(unit_act='tanh',r_prior = 'kurtotic', U_prior = 'kurtotic', input_size=576,
    #     hidden_sizes = [96,32], num_epochs = 40,
    #     k_r_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}},
    #     k_U_sched = {'step':{'initial':0.005,'drop_factor':0.9,'drop_every':10}},
    #     k_o_sched = {'step':{'initial':0.05,'drop_factor':0.9,'drop_every':10}})


    """
    PCMOD OBJECT NAMING FORMAT
    and pipeline instructions
    """

    # every pickled file in the main -> evaluation -> prediction -> plotting script pipeline
    # has been named with the following format:

    # pc . [sizelayer1.sizelayer2] . transform function . prior . classification method
    # . trained or not . # epochs if trained . training image set
    # . evaluated or not . evaluation image set . predicted with or not . prediction image set . extra tag . pydb

    # e.g.
    # pc.[32.10].tanh.gauss.C2.T.100e.tanh100x10.ne.-.np.-.randUo.pydb

    # the above model:
    # is size 32,10, tanh, gaussian, has C2 classification, was trained for 100 epochs on
    # the tanh 100x10 dataset, was not evaluated (ne) on any evaluation image set (-), was not
    # used for prediction (np) on any images (-), and its extra tag indicates its train() loop in model.py
    # was set so that Uo would stay random, never updating based on the labels (hence: 'randUo')

    # note that "-" serves as a placeholder for "not present"

    # when commenting-in the correct naming parameters for pickling in and out in main, eval, pred, or plot,
    # note that they must be strings. I hope to set this up at some point as a function that runs with less human input.

    # though the names are backwards compatible, currently the scripts can only be run in one direction

    # main -> evaluation -> prediction -> plotting
    # main -> prediction -> plotting
    # main -> evaluation -> plotting
    # main -> plotting

    # note that evaluation and/or prediction can be skipped


    """
    Set Naming Parameters for Model (Define Model Output Pickle Filename)
    MUST comment-in desired naming parameters
    MUST match model parameters set above
    """

    # pickle output model
    # MUST comment-in desired names of parameters in the model

    # "-" serves as a placeholder for "not present"

    #transformation function
    transform_type = 'tanh'
    # transform_type = 'linear'

    #prior type
    prior_type = 'gauss'
    # prior_type = 'kurt'

    #classification method
    # class_type = 'NC'
    class_type = 'C1'
    # class_type = 'C2'

    #will be trained or untrained
    trained = 'T'
    # trained = 'nt'

    #number of epochs if trained (if not, use -)
    # num_epochs = '1000e'
    # num_epochs = '200e'
    # num_epochs = '100e'
    # num_epochs = '50e'
    # num_epochs = '40e'
    # num_epochs = '25e'
    # num_epochs = '20e'
    num_epochs = '10e'
    # num_epochs = '-'

    #dataset trained on if trained (if not, use -)
    # training_dataset = 'tanh100x10'
    # training_dataset = 'tanh1000x10'
    training_dataset = 'tanh100x10_size_24x24'
    # training_dataset = 'linear100x10_size_24x24'
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
    # extra_tag = 'pipeline_test'
    # extra_tag = 'tile_offset_6_lr_0.5_lU_0.005'
    # extra_tag = 'tile_offset_6_lr_0.5_lU_0.0005'
    # extra_tag = 'tile_offset_6_lr_0.05_lU_0.001'
    # extra_tag = 'tile_offset_6_lr_0.05_lU_0.005'
    # extra_tag = 'tile_offset_6_lr_0.005_lU_0.005'
    # extra_tag = 'tile_offset_6_lr_0.0005_lU_0.0005'
    # extra_tag = 'tile_offset_6_poly_0.05_pp1'
    # extra_tag = 'tile_offset_6_poly_0.005_pp1'
    # extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
    # extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1'
    # extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1_randUo'
    # extra_tag = 'tile_offset_6_const_lr_0.05'
    # extra_tag = 'poly_lr_0.05_lU_0.005_me40_pp1'
    # extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me20_pp1'
    # extra_tag = 'tile_offset_6_step_0.005_df0.9_10'
    # extra_tag = 'tile_offset_6_step_lr_0.05_lU_0.005_df0.9_10'
    # extra_tag = 'const_lr_0.0005'
    # extra_tag = 'tile_offset_6'
    # extra_tag = 'tile_offset_8'
    # extra_tag = 'tile_offset_0'
    # extra_tag = 'cboost_1'
    # extra_tag = 'cboost_5'
    # extra_tag = 'cboost_50'
    # extra_tag = 'cboost_100'
    # extra_tag = 'cboost_1000'
    # extra_tag = 'cboost_4000'
    extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1'
    # extra_tag = '-'

    """
    Pickle In Training Image Set
    comment-in correct image set
    """

    # load preprocessed data saved by preprocessing.py
    # for a linear model training on a linear-optimized training set of 10digs x 10imgs open "linear_10x10.pydb"
    # comment out the below three lines if using tanh model

    # linear_tile_data_in = open('linear_100x10_size_24x24.pydb','rb')
    # X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(linear_tile_data_in)
    # linear_tile_data_in.close()

    # for a tanh model training on a tanh-optimized training set of 10digs x 10imgs open "tanh_10x10.pydb"
    # for 100x10, use the associated tanh_100x10.pydb
    # for 1000x10, use the associated tanh_1000x10.pydb
    # comment out the below three lines if using linear model

    tanh_tile_data_in = open('tanh_100x10_size_24x24.pydb','rb')
    X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_tile_data_in)
    tanh_tile_data_in.close()



    """
    Train
    """

    # list naming parameters above: if anything left unset, train() will not run
    naming_parameters = [model_size,transform_type,prior_type,class_type,\
        trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag]

    # NOTE: comment out pcmod.train() line below to leave model untrained

    # # # instantiate model
    # pcmod = PredictiveCodingClassifier(p)
    # # train on training set
    # pcmod.train(X_train, y_train)

    # instantiate and train tiled model
    tiled_pcmod = TiledPredictiveCodingClassifier(p)

    # safeguard against training a non-tiled model
    if tiled_pcmod.is_tiled == True:
        tiled_pcmod.train(X_train, y_train)


    """
    Pickle Out Trained or Untrained Model
    """

    # # pickle output model
    #
    # pcmod_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
    #     trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
    # pickle.dump(pcmod, pcmod_out)
    # pcmod_out.close()

    # pickle tiled output model

    tiled_pcmod_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
        trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
    pickle.dump(tiled_pcmod, tiled_pcmod_out)
    tiled_pcmod_out.close()



if __name__ == '__main__':
    # for unabridged cProfile readout in bash shell type: 'python -m cProfile main.py'

    time_start = datetime.datetime.now
    main()
    time_end = datetime.datetime.now
    time_taken = time_end-time_start

    print('time start {}'.format(time_start))
    print('time end {}'.format(time_end))
    print('time taken {}'.format(time_taken))

    # for truncated cProfile readout in IDE, use logic below

    # NOTE: fix this
    # pst = pstats.Stats('restats')
    # pst.strip_dirs().sort_stats(-1).print_stats()
    # cProfile.run('main()')
