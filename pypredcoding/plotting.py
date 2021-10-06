"""
Script for plotting data from trained PredictiveCodingClassifier object
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from kbutil import tsne


"""
Comment-in the plot type
6 options
"""


# plot_type = 'trainingE'

# # for models with different E,C scales
# plot_type = 'trainingECAsplitplot'
# 
# # for models with same E,C scale
plot_type = 'trainingECAoneplot'

# # for models with different E,C scales
# plot_type = 'evalECAsplitplot'

# # for models with same E,C scale
# plot_type = 'evalECAoneplot'

# plot_type = 'predPEs'


"""
Comment-in the naming parameters of the object being plotted
MUST comment-in 'T' and a training_dataset for trainingE, trainingECA split, and trainingECAone plots
MUST comment-in 'E' and an eval_dataset for evalECAsplit and evalECAone plots
MUST comment-in 'P' and a pred_dataset for predPEs plot
"""


#model size
# model_size = '[32.10]'
# model_size = '[32.32]'
model_size = '[128.36]'
# model_size = '[36.32]'
# model_size = '[36.10]'
# model_size = '[288.10]'
# model_size = '[2304.10]'
# model_size = '[18432.10]'
# model_size = '[36.36]'
# model_size = '[288.288]'
# model_size = '[576.576]'
# model_size = '[1152.1152]'
# model_size = '[1152.10]'
# model_size = '[2304.2304]'
# model_size = '[18432.18432]'
# model_size = '[1280.10]'
# model_size = '[128.10]'
# model_size = '[128.128.10]'
# model_size = '[128.128.128.10]'
# model_size = '[128.128.128.128.10]' 
# model_size = '[1020.10]'
# model_size = '[128.32]'
# model_size = '[96.32]'
# model_size = '[192.32]'



#transformation function
transform_type = 'tanh'
# transform_type = 'linear'

#prior type
prior_type = 'gauss'
# prior_type = 'kurt'

#classification method
class_type = 'NC'
# class_type = 'C1'
# class_type = 'C2'

#trained or untrained
trained = 'T'
# trained = 'nt'

#number of epochs if trained (if not, use -)
# num_epochs = '10000e'
# num_epochs = '5000e'
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
training_dataset = 'tanh100x10'
# training_dataset = 'tanh_dog_10x10'
# training_dataset = 'tanh1000x10'
# training_dataset = 'tanh100x10_size_24x24'
# training_dataset = 'linear100x10_size_24x24'
# training_dataset = 'tanh10x10'
# training_dataset = 'tanh1x10'
# training_dataset = 'rao_visionres'
# training_dataset = 'rao_ballard_nature'
# training_dataset = 'rao_visionres_size_24x24'
# training_dataset = 'rao_ballard_nature_size_24x24'
# training_dataset = '-'

#evaluated or not evaluated with so far
# evaluated = 'E'
evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
# eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh100x10_fashion_mnist'
# eval_dataset = 'tanh100x10_cifar10'
# eval_dataset = 'tanh100x10_size_24x24'
# eval_dataset = 'tanh100x10_fashion_mnist_size_24x24'
# eval_dataset = 'tanh100x10_cifar10_size_24x24'
# eval_dataset = 'tanh10x10'
# eval_dataset = 'ten_of_each_dig_from_mnist_1000'
eval_dataset = '-'


#used or not used for prediction so far
# used_for_pred = 'P'
used_for_pred = 'np'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
# pred_dataset = '5imgs'
# pred_dataset = '0-9_minE_128.32_kurt'
# pred_dataset = '0-9_maxE_128.32_kurt'
pred_dataset = '-'

#extra identifier for any particular or unique qualities of the model object
# extra_tag = 'randUo'
# extra_tag = 'scaled_ppixel'
# extra_tag = 'pipeline_test'
# extra_tag = 'tile_offset_6'
# extra_tag = 'tile_offset_6_lr_0.5_lU_0.005'
# extra_tag = 'tile_offset_6_lr_0.5_lU_0.0005'
# extra_tag = 'tile_offset_6_lr_0.05_lU_0.001'
# extra_tag = 'tile_offset_6_lr_0.05_lU_0.005'
# extra_tag = 'tile_offset_6_lr_0.005_lU_0.005'
# extra_tag = 'tile_offset_6_lr_0.0005_lU_0.0005'
# extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
# extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1_randUo'
# extra_tag = 'poly_lr_0.05_lU_0.005_me40_pp1'
# extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me20_pp1'
# extra_tag = 'tile_offset_6_poly_0.05_pp1'
# extra_tag = 'tile_offset_6_step_lr_0.05_lU_0.005_df0.9_10'
# extra_tag = 'tile_offset_6_step_0.005_df0.9_10'
# extra_tag = 'const_lr_0.0005'
# extra_tag = 'cboost_1'
# extra_tag = 'cboost_5'
# extra_tag = 'cboost_50'
# extra_tag = 'cboost_100'
# extra_tag = 'cboost_1000'
# extra_tag = 'cboost_4000'
# extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me10_pp1'
# extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1'
# extra_tag = 'tiled'
# extra_tag = 'C2_no_L'  
# extra_tag = 'C2_LSQ'
# extra_tag = 'C1_LSQ'
extra_tag = '-'


"""
Define plotting function
"""

# NOTE: change plots or add new plots inside this function
# if you make a new plot, add a new plot_type and describe it under the plot type docstring above

def plot(plot_type,model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag):

    #NOTE: will go through and standardize plotting variable names at some point

    if plot_type == 'trainingE':
        
        if trained != "T":
            print('model must be trained (trained = "T") to run "trainingE" plot')
            
        elif trained == 'T':

            """
            PLOTTING TRAINING Loss (E only) of a Classifying or Non-Classifying Model
            """
    
            print("Plot training loss (E) of a classifying or non-classifying model")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
            """
            Pickle In
            """
    
            # load it
            pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod = pickle.load(pcmod_in)
            pcmod_in.close()
    
            """
            Set Variables
            """
    
            # model variables
            transform_function = pcmod.p.unit_act
            classification_type_during_training = pcmod.class_type
            prior_type = pcmod.p.r_prior
            representation_cost_by_epoch = pcmod.E_avg_per_epoch
            lr_r = pcmod.lr_r
            lr_U = pcmod.lr_U
            num_epochs = pcmod.p.num_epochs
            num_epochs = range(1,num_epochs+1)
            
            # # tiling stuff
            # is_tiled = pcmod.is_tiled
            # tile_offset = pcmod.p.tile_offset
            
            # if is_tiled == True:
            #     tiling = "tile_offset {}".format(tile_offset)
            # else:
            #     tiling = " "
    
    
            # loss after completion of one epoch
            E1 = round(representation_cost_by_epoch[1],2)
            # loss min
            Emin = round(min(representation_cost_by_epoch),2)
    
            """
            Plot
            """
    
            # set title
            fig, ax = plt.subplots(1)
            fig.suptitle("{}  {}  {}   ".format(transform_function,prior_type,classification_type_during_training)+\
            "lr_r={} ".format(lr_r)+"lr_U={} ".format(lr_U)+'\n'\
            +'E1={} '.format(E1)+'Emin={} '.format(Emin))
    
            # set color
            plotE = ax.plot(num_epochs, representation_cost_by_epoch, '#000000', label="Avg E")
    
            # set E scale
            ax.set_ylim(0, 2000)
    
            # set axis names
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg E")
    
            # show plot
            plt.show()
            
        else:
            print("attempted to plot this object with trainingE: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("trained must = 'T' or 'nt'")
            print("train model first [main.py]")


    elif plot_type == 'trainingECAsplitplot':
        
        if trained != "T":
            print('model must be trained (trained = "T") to run "trainingECAsplitplot" plot')
            
        elif trained == 'T':

            """
            PLOTTING TRAINING Loss (E), Classification Loss (C) and Accuracy (A) of a Classifying Model
    
            DIFFERENT E,C SCALES
    
            Generates Two Subplots: E & A on the top, and C & A on the bottom
            """
    
            print("Plot training E,C,A of a classifying model with E,A on top, C,A on bottom")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
    
            """
            Pickle In
            """
    
            # load it
            pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod = pickle.load(pcmod_in)
            pcmod_in.close()
    
    
            """
            Set Variables
            """
    
            # E
            round_first = round(pcmod.E_avg_per_epoch[0],2)
            round_epoch1 = round(pcmod.E_avg_per_epoch[1],2)
            round_last = round(pcmod.E_avg_per_epoch[-1],2)
            round_min = round(min(pcmod.E_avg_per_epoch),2)
    
            # C
            C_round_first = round(pcmod.C_avg_per_epoch[0],2)
            C_round_last = round(pcmod.C_avg_per_epoch[-1],2)
            C_round_min = round(min(pcmod.C_avg_per_epoch),2)
            C_round_max = round(max(pcmod.C_avg_per_epoch),2)
    
            # E+C
            Eavg_plus_Cavg_per_epoch = pcmod.E_avg_per_epoch + pcmod.C_avg_per_epoch
            EC_first = round_first + C_round_first
            EC_last = round_last + C_round_last
            EC_min = round_min + C_round_min
    
            # Accuracy
            acc_first = round(pcmod.acc_per_epoch[0],1)
            acc_last = round(pcmod.acc_per_epoch[-1],1)
            acc_max = round(max(pcmod.acc_per_epoch),1)
            acc_avg = round(pcmod.acc_per_epoch,1)
    
    
            # general variables
            num_epochs = range(1, pcmod.p.num_epochs+1)
            representation_cost = pcmod.E_avg_per_epoch
            classification_cost = pcmod.C_avg_per_epoch
            accuracy = pcmod.acc_per_epoch
            class_type = pcmod.class_type
            prior_type = pcmod.p.r_prior
            
            # # tiling stuff
            # is_tiled = pcmod.is_tiled
            # tile_offset = pcmod.p.tile_offset
            
            # if is_tiled == True:
            #     tiling = "tile_offset {}".format(tile_offset)
            # else:
            #     tiling = " "
                
            tiling = " "
    
    
            """
            Plot
            """
    
            # plot E/Acc vs epoch; plot C/Acc vs epoch
    
            # split into vertically-stacked subplots
    
            fig, (axE, axC) = plt.subplots(2)
            fig.suptitle("{}  {}  {}  {}  ".format(pcmod.p.unit_act,prior_type,class_type,tiling)+"lr_r={} ".format(pcmod.lr_r)+"lr_U={} ".format(pcmod.lr_U)+"lr_o={}".format(pcmod.lr_o)+'\n'\
            +'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
            +'Cmax={} '.format(C_round_max)+'Cmin={} '.format(C_round_min) + 'Aavg={} '.format(acc_avg))
    
            # create a second y-axis (Accuracy) for each subplot
    
            twinEA = axE.twinx()
            twinCA = axC.twinx()
    
            # create labeled plot objects
            # black, sky,
            plotE = axE.plot(num_epochs, representation_cost, '#000000', label="Avg E")
            plotC = axC.plot(num_epochs, classification_cost, '#4363d8', label="Avg C")
            plotEA = twinEA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")
            plotCA = twinCA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")
    
            # set limits for and label x,y-axes for both subplots
    
            # axE.set_xlim(0, 2)
            axC.set_ylim(0, 0.07)
            axE.set_ylim(0, 1)
            twinEA.set_ylim(0, 100)
            twinCA.set_ylim(0, 100)
    
            axE.set_xlabel("Epoch")
            axE.set_ylabel("Avg E")
            twinEA.set_ylabel("Accuracy")
    
            axC.set_xlabel("Epoch")
            axC.set_ylabel("Avg C")
            twinCA.set_ylabel("Accuracy")
    
    
            # show plot
            plt.show()
            
            
        else:
            print("attempted to plot this object with trainingECAsplitplot: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("trained must = 'T' or 'nt'")
            print("train model first [main.py]")
    



    elif plot_type == 'trainingECAoneplot':
        
        if trained != "T":
            print('model must be trained (trained = "T") to run "trainingECAoneplot" plot')
            
        elif trained == 'T':

            """
            PLOTTING TRAINING Loss (E), Classification Loss (C) and Accuracy (A) of a Classifying Model
    
            SAME E,C SCALE
    
            Generates One Plot: E, C & A Together
            """
    
            print("Plot training E,C,A of a classifying model with E,C,A together in one plot")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
            """
            Pickle In
            """
    
            # load it
            pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod = pickle.load(pcmod_in)
            pcmod_in.close()
    
    
            """
            Set Variables
            """
    
            # E
            round_first = round(pcmod.E_avg_per_epoch[0],3)
            round_epoch1 = round(pcmod.E_avg_per_epoch[1],3)
            round_last = round(pcmod.E_avg_per_epoch[-1],3)
            round_min = round(min(pcmod.E_avg_per_epoch),3)
    
            # C
            C_round_first = round(pcmod.C_avg_per_epoch[0],3)
            C_round_last = round(pcmod.C_avg_per_epoch[-1],3)
            C_round_min = round(min(pcmod.C_avg_per_epoch),3)
            C_round_max = round(max(pcmod.C_avg_per_epoch),3)
    
            # E+C
            Eavg_plus_Cavg_per_epoch = pcmod.E_avg_per_epoch + pcmod.C_avg_per_epoch
            EC_first = round_first + C_round_first
            EC_last = round_last + C_round_last
            EC_min = round_min + C_round_min
    
            # Accuracy
            acc_first = round(pcmod.acc_per_epoch[0],1)
            acc_last = round(pcmod.acc_per_epoch[-1],1)
            acc_max = round(max(pcmod.acc_per_epoch),1)
            acc_avg = round(np.mean(pcmod.acc_per_epoch),1)
    
    
            # general variables
            num_epochs = range(1, pcmod.p.num_epochs+1)
            representation_cost = pcmod.E_avg_per_epoch
            classification_cost = pcmod.C_avg_per_epoch
            accuracy = pcmod.acc_per_epoch
            class_type = pcmod.class_type
            prior_type = pcmod.p.r_prior
            
            print("classification cost per epoch")
            print(classification_cost)
            print("accuracy")
            print(accuracy)
            
            # # tiling stuff
            # is_tiled = pcmod.is_tiled
            # tile_offset = pcmod.p.tile_offset
            
            # if is_tiled == True:
            #     tiling = "tile_offset {}".format(tile_offset)
            # else:
                # tiling = " "
                
            tiling = ' '
    
    
            """
            Plot
            """
    
            fig, (axE) = plt.subplots(1)
            fig.suptitle("{}  {}  {}  {}  {}".format(pcmod.p.unit_act,prior_type,class_type,tiling,pcmod.p.hidden_sizes)+'\n'\
            +'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
            +'Cmax={} '.format(C_round_max)+'Cmin={} '.format(C_round_min) + 'Aavg={} '.format(acc_avg))
    
    
            twinEA = axE.twinx()
    
            # set colors
            plotE = axE.plot(num_epochs, representation_cost, '#000000', label="Avg E")
            plotC = axE.plot(num_epochs, classification_cost, '#4363d8', label="Avg C")
            plotEA = twinEA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")
    
            # set limits for and label x,y-axes for both subplots
    
            # axE.set_xlim(0, 2)
            axE.set_ylim(0, 20000)
            twinEA.set_ylim(0, 110)
    
            axE.set_xlabel("Epoch")
            axE.set_ylabel("Avg E, Avg C")
            twinEA.set_ylabel("Accuracy")
    
            # show plot
            plt.show()
            
        else:
            print("attempted to plot this object with trainingECAoneplot: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("trained must = 'T' or 'nt'")
            print("train model first [main.py]")



    elif plot_type == 'evalECAsplitplot':

        """
        PLOTTING EVALUATION E, C, and Acc of a Trained or Untrained, Classifying or Non-Classifying Model

        DIFFERENT E,C SCALES

        Generates Two Subplots: E & A on the top, and C & A on the bottom
        """
        
        if evaluated != "E":
            print('model must be evaluated (evaluated = "E") to run "evalECAsplitplot" plot')
            
        elif evaluated == 'E':

            print("Plot evaluation E,C,A of a classifying or non-classifying model with E,A on top, C,A on bottom")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
    
            """
            Pickle In
            """
    
            # load it
            evaluation_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod,E,C,Classif_success_by_img,Acc,softmax_guess_each_img = pickle.load(evaluation_in)
            evaluation_in.close()
    
            """
            Set Variables
            """
    
            n_eval_images = pcmod.n_eval_images
            n_eval_images = range(1,n_eval_images+1)
    
            # # double check that evaluation objects contain the right values
            # print(E)
            # print(C)
            # print(Classif_success_by_img)
            # print(Acc)
    
            # E
    
            Eavg = round((sum(E)/pcmod.n_eval_images),2)
            Emin = round(min(E),2)
    
            # C
            Cavg = round((sum(C)/pcmod.n_eval_images),2)
            Cmin = round(min(C),2)
    
    
            # Acc
            Acc = round(Acc,2)
    
            # general variables
            eval_class_type = pcmod.eval_class_type
            prior_type = pcmod.p.r_prior
            
            # # tiling stuff
            # is_tiled = pcmod.is_tiled
            # tile_offset = pcmod.p.tile_offset
            
            # if is_tiled == True:
            #     tiling = "tile_offset {}".format(tile_offset)
            # else:
            #     tiling = " "
                
            tiling = " "
            
            """
            Plot
            """
    
    
            fig, (axE, axC) = plt.subplots(2)
            fig.suptitle("{}  {}  {} eval classif type={}     ".format(pcmod.p.unit_act,prior_type,tiling,eval_class_type)+'\n'\
            +'Eavg={} '.format(Eavg)+'Emin={} '.format(Emin)+ 'Cavg={} '.format(Cavg) +'Cmin={} '.format(Cmin)+ 'Acc={} '.format(Acc))
    
    
            # create labeled plot objects
            # black, sky,
            plotE = axE.plot(n_eval_images, E, '#000000', label="E")
            plotC = axC.plot(n_eval_images, C, '#4363d8', label="C")
    
    
            # set limits for and label x,y-axes for both subplots
    
            # E plotting range is around 2000 if model is untrained; if trained on 100 images, E ~ 10; if trained on 1000 images, E ~ 1.
            axE.set_ylim(0, 2000)
            # C plotting range is around 20 if model is untrained; if trained on 100 images, E ~ 0.15; if trained on 1000 images, E ~ 0.05.
            axC.set_ylim(0, 100)
    
    
            axE.set_xlabel("Evaluation Image")
            axE.set_ylabel("E")
    
    
            axC.set_xlabel("Evaluation Image")
            axC.set_ylabel("C")
    
    
            # show plot
            plt.show()
            
        else:
            print("attempted to plot this object with evalECAsplitplot: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("evaluated must = 'E' or 'ne'")
            print("evaluate model first [evaluation.py]")



    elif plot_type == 'evalECAoneplot':

        """
        Plotting EVALUATION E, C, and Acc of a Trained or Untrained, Classifying or Non-Classifying Model

        SAME E,C SCALE

        Generates One Plot: E, C & A Together
        """
        
        if evaluated != "E":
            print('model must be evaluated (evaluated = "E") to run "evalECAsplitplot" plot')
            
        elif evaluated == 'E':

            print("Plot evaluation E,C,A of a classifying model with E,C,A together in one plot")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
    
            """
            Pickle In
            """
    
            # load it
            evaluation_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod,E,C,Classif_success_by_img,Acc = pickle.load(evaluation_in)
            evaluation_in.close()
    
    
            """
            Set Variables
            """
    
            n_eval_images = pcmod.n_eval_images
            n_eval_images = range(1,n_eval_images+1)
    
    
            # E,C and Accuracy data points for plotting
    
            # E
    
            # # double check that evaluation objects contain the right values
            # print(E)
            # print(C)
            # print(Classif_success_by_img)
            # print(Acc)
    
            Eavg = round((sum(E)/pcmod.n_eval_images),2)
            Emin = round(min(E),2)
    
            # C
            Cavg = round((sum(C)/pcmod.n_eval_images),2)
            Cmin = round(min(C),2)
    
            # Acc
            Acc = Acc
    
    
            # general variables
            eval_class_type = pcmod.eval_class_type
            prior_type = pcmod.p.r_prior
            
            # tiling stuff
            is_tiled = pcmod.is_tiled
            tile_offset = pcmod.p.tile_offset
            
            if is_tiled == True:
                tiling = "tile_offset {}".format(tile_offset)
            else:
                tiling = " "
    
    
            """
            Plot
            """
    
            fig, (axE) = plt.subplots(1)
            fig.suptitle("{}  {}  {} eval_class_type={}  {}  ".format(pcmod.p.unit_act,prior_type,tiling,eval_class_type,pcmod.p.hidden_sizes)+'\n'\
            +'Eavg={} '.format(Eavg)+'Emin={} '.format(Emin)\
            +'Cavg={} '.format(Cavg)+'Cmin={} '.format(Cmin) + 'Acc={} '.format(Acc))
    
    
            # set colors
            plotE = axE.plot(n_eval_images, E, '#000000', label="E")
            plotC = axE.plot(n_eval_images, C, '#4363d8', label="C")
    
            # set limits for and label x,y-axes for both subplots
    
            # E/C plotting range is around 2000 if model is untrained; if trained on 100 images, E/C ~~ 10; if trained on 1000 images, E/C ~~ 1.
            axE.set_ylim(0, 2000)
    
    
            axE.set_xlabel("Evaluation Image")
            axE.set_ylabel("E, C")
    
            # show plot
            plt.show()
            
        else:
            print("attempted to plot this object with evalECAoneplot: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("evaluated must = 'E' or 'ne'")
            print("evaluate model first [evaluation.py]")


    elif plot_type == 'predPEs':

        """
        PLOTTING PREDICTION Errors (L1 & L2) of a Trained or Untrained Model

        Sequentially Generates One PE Plot Per Image Predicted
        """
        
        if used_for_pred != "P":
            print('model must have been used for prediction (used_for_pred = "P") to run "predPEs" plot')
            
        elif used_for_pred == 'P':

            print("Sequentially generate one PE Plot (with L1,L2 PEs) per prediction image ")
            print("Object plotted: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
    
            """
            Pickle In
            """
    
            # import model and predicted image set (set could contain multple 2-dim images, or one 3-dim multi-image
            # vector, like X_train)
            prediction_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
            pcmod, prediction_image_set, n_pred_images = pickle.load(prediction_in)
            prediction_in.close()
    
    
            """
            Set Variables
            """
    
            # model variables
            transform_function = pcmod.p.unit_act
            classification_type_during_training = pcmod.class_type
            prior_type = pcmod.p.r_prior
            num_updates = range(1,pcmod.n_pred_updates+1)
            
            # tiling stuff
            is_tiled = pcmod.is_tiled
            tile_offset = pcmod.p.tile_offset
            
            if is_tiled == True:
                tiling = "tile_offset {}".format(tile_offset)
            else:
                tiling = " "
    
    
            print(pcmod.prediction_errors_l1[0])
            print(pcmod.prediction_errors_l1[0][-1])
            print(pcmod.n_pred_images)
    
    
            """
            Plot
            """
    
    
            for prediction_image in range(0,n_pred_images):
    
                # prediction specific variables
    
                pe_1_first = round(pcmod.prediction_errors_l1[prediction_image][0],1)
                pe_1_last = round(pcmod.prediction_errors_l1[prediction_image][-1],1)
                pe_2_first = round(pcmod.prediction_errors_l2[prediction_image][0],1)
                pe_2_last = round(pcmod.prediction_errors_l2[prediction_image][-1],1)
    
    
                fig, ax = plt.subplots(1)
                fig.suptitle("{}  {}  {}  {} pred_img {}  ".format(pcmod.p.unit_act,pcmod.class_type,prior_type,tiling,prediction_image+1)+'\n'\
                +'pe_1_first={} '.format(pe_1_first)+'pe_1_last={} '.format(pe_1_last)\
                + 'pe_2_first={} '.format(pe_2_first) + 'pe_2_last={} '.format(pe_2_last))
    
    
                # black and navy
                plotE = ax.plot(num_updates, pcmod.prediction_errors_l1[prediction_image], '#000000', label="pe_1")
                plotE = ax.plot(num_updates, pcmod.prediction_errors_l2[prediction_image], '#000075', label="pe_2")
                ax.set_ylim(0, 50)
                ax.legend()
    
                ax.set_xlabel("Update")
                ax.set_ylabel("L1, L2 PE")
    
    
                plt.show()
                
        else:
            print("attempted to plot this object with predPEs: ")
            print('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
              trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag))
            print("used_for_pred must = 'P' or 'np'")
            print("predict image(s) with this model first [prediction.py]")
        
        
        
        
    else:
        print("plot type must be one of pre-listed six types")




"""
Run plotting function
this shouldn't be changed often, if ever
"""


plot(plot_type,model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag)




"""
STORAGE
"""

"""
Colors from kbutil and matplotlib
"""

# matplotlib default colors
_colors = ('k','r','orange','gold','g','b','purple','magenta',
           'firebrick','coral','limegreen','dodgerblue','indigo','orchid',
           'tomato','darkorange','greenyellow','darkgreen','yellow','deepskyblue','indigo','deeppink')

# colorwheel colors
_colors = ('#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
            '#f032e6', '#bcf60c','#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#ffe119')

# br's unofficial names of colorwheel colors

lime_green='#3cb44b'
magenta_red='#e6194b'
sky_blue='#4363d8'
light_orange='#f58231'
bright_purple='#911eb4'
turquoise='#46f0f0'
magenta_pink='#f032e6'
neon_yellow_green='#bcf60c'
pink_skin='#fabebe'
dark_lakefoam_green='#008080'
powder_pink='#e6beff'
popcorn_yellow='#fffac8'
carmel_brown='#9a6324'
dark_chocolate='#800000'
keylime_pie='#aaffc3'
camo_green='#808000'
cooked_salmon='#ffd8b1'
navy_blue='#000075'
dark_grey='#808080'
white='#ffffff'
black='#000000'
gold_yellow:'#ffe119'


# example of plt.annotate method

#     plt.annotate("E avg initial = {}".format(round_first) + '\n' \
#     + "E avg final = {}".format(round_last) + '\n' \
#     + "E avg min = {}".format(round_min) + '\n' \
#     + "E avg total descent = {}".format(round((round_first - round_last),1)), (0.58,0.67), xycoords='figure fraction')
