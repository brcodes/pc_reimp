"""
Script for plotting data from trained PredictiveCodingClassifier object
"""

"""
(comment-in the text below each "Plotting" docstring, and run, to generate plots)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle



"""
Plotting TRAINING Loss (E only) of a Classifying or Non-Classifying Model
"""

# # load trained model

# prior_type = 'gauss'
# # prior_type = 'kurt'

# class_type = 'NC'
# # class_type = 'C1'
# # class_type = 'C2'


# pcmod_in = open('pcmod_trained_1000imgs_{}_{}.pydb'.format(prior_type,class_type),'rb')
# pcmod = pickle.load(pcmod_in)
# pcmod_in.close()


# # model variables
# transform_function = pcmod.p.unit_act
# classification_type_during_training = pcmod.class_type
# prior_type = pcmod.p.r_prior
# representation_cost_by_epoch = pcmod.E_avg_per_epoch
# lr_r = pcmod.lr_r
# lr_U = pcmod.lr_U
# num_epochs = pcmod.p.num_epochs
# num_epochs = range(1,num_epochs+1)


# # loss after completion of one epoch
# E1 = round(representation_cost_by_epoch[1],2)
# # loss min
# Emin = round(min(representation_cost_by_epoch),2)

# # set title 
# fig, ax = plt.subplots(1)
# fig.suptitle("{}  {}  {}     ".format(transform_function,prior_type,classification_type_during_training)+\
# "lr_r={} ".format(lr_r)+"lr_U={} ".format(lr_U)+'\n'\
# +'E1={} '.format(E1)+'Emin={} '.format(Emin))


# # set color
# plotE = ax.plot(num_epochs, representation_cost_by_epoch, '#000000', label="Avg E")

# # set E scale
# ax.set_ylim(0, 0.7)

# # set axis names
# ax.set_xlabel("Epoch")
# ax.set_ylabel("Avg E")

# # show plot
# plt.show()


"""
Plotting TRAINING Loss (E), Classification Loss (C) and Accuracy (A) of a Classifying Model

DIFFERENT SCALES for E and C

Generates Two Subplots: E & A on the top, and C & A on the bottom
"""


# # load trained model as a pickle

# prior_type = 'gauss'
# # prior_type = 'kurt'

# # class_type = 'NC'
# # class_type = 'C1'
# class_type = 'C2'

# pcmod_in = open('pcmod_trained_1000imgs_{}_{}.pydb'.format(prior_type,class_type),'rb')
# pcmod = pickle.load(pcmod_in)
# pcmod_in.close()

# # display total number of model parameters after training

# print('Total number of model parameters')
# print(pcmod.n_model_parameters)
# print('\n')

# # E,C and Accuracy data points for plotting

# # E
# round_first = round(pcmod.E_avg_per_epoch[0],2)
# round_epoch1 = round(pcmod.E_avg_per_epoch[1],2)
# round_last = round(pcmod.E_avg_per_epoch[-1],2)
# round_min = round(min(pcmod.E_avg_per_epoch),2)

# # C
# C_round_first = round(pcmod.C_avg_per_epoch[0],2)
# C_round_last = round(pcmod.C_avg_per_epoch[-1],2)
# C_round_min = round(min(pcmod.C_avg_per_epoch),2)
# C_round_max = round(max(pcmod.C_avg_per_epoch),2)

# # E+C
# Eavg_plus_Cavg_per_epoch = pcmod.E_avg_per_epoch + pcmod.C_avg_per_epoch
# EC_first = round_first + C_round_first
# EC_last = round_last + C_round_last
# EC_min = round_min + C_round_min

# # Accuracy
# acc_first = round(pcmod.acc_per_epoch[0],1)
# acc_last = round(pcmod.acc_per_epoch[-1],1)
# acc_max = round(max(pcmod.acc_per_epoch),1)


# # general variables
# num_epochs = range(1, pcmod.p.num_epochs+1)
# representation_cost = pcmod.E_avg_per_epoch
# classification_cost = pcmod.C_avg_per_epoch
# accuracy = pcmod.acc_per_epoch
# class_type = pcmod.class_type
# prior_type = pcmod.p.r_prior

# # plot E/Acc vs epoch; plot C/Acc vs epoch

# # split into vertically-stacked subplots

# fig, (axE, axC) = plt.subplots(2)
# fig.suptitle("{}  {}  {}     ".format(pcmod.p.unit_act,prior_type,class_type)+"lr_r={} ".format(pcmod.lr_r)+"lr_U={} ".format(pcmod.lr_U)+"lr_o={}".format(pcmod.lr_o)+'\n'\
# +'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
# +'Cmax={} '.format(C_round_max)+'Cmin={} '.format(C_round_min) + 'Amax={} '.format(acc_max))

# # create a second y-axis (Accuracy) for each subplot

# twinEA = axE.twinx()
# twinCA = axC.twinx()

# # create labeled plot objects
# # black, sky,
# plotE = axE.plot(num_epochs, representation_cost, '#000000', label="Avg E")
# plotC = axC.plot(num_epochs, classification_cost, '#4363d8', label="Avg C")
# plotEA = twinEA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")
# plotCA = twinCA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")

# # set limits for and label x,y-axes for both subplots

# # axE.set_xlim(0, 2)
# axC.set_ylim(0, 0.07)
# axE.set_ylim(0, 1)
# twinEA.set_ylim(0, 100)
# twinCA.set_ylim(0, 100)

# axE.set_xlabel("Epoch")
# axE.set_ylabel("Avg E")
# twinEA.set_ylabel("Accuracy")

# axC.set_xlabel("Epoch")
# axC.set_ylabel("Avg C")
# twinCA.set_ylabel("Accuracy")


# # show plot
# plt.show()



"""
Plotting TRAINING Loss (E), Classification Loss (C) and Accuracy (A) of a Classifying Model

SAME SCALE for E and C

Generates One Plot: E, C & A Together
"""


# # load trained model as a pickle
# prior_type = 'gauss'
# # prior_type = 'kurt'

# # class_type = 'NC'
# # class_type = 'C1'
# class_type = 'C2'

# pcmod_in = open('pcmod_trained_1000imgs_100eps_{}_{}.pydb'.format(prior_type,class_type),'rb')
# pcmod = pickle.load(pcmod_in)
# pcmod_in.close()

# # display total number of model parameters after training

# print('Total number of model parameters')
# print(pcmod.n_model_parameters)
# print('\n')

# # E,C and Accuracty data points for plotting

# # E
# round_first = round(pcmod.E_avg_per_epoch[0],3)
# round_epoch1 = round(pcmod.E_avg_per_epoch[1],3)
# round_last = round(pcmod.E_avg_per_epoch[-1],3)
# round_min = round(min(pcmod.E_avg_per_epoch),3)

# # C
# C_round_first = round(pcmod.C_avg_per_epoch[0],3)
# C_round_last = round(pcmod.C_avg_per_epoch[-1],3)
# C_round_min = round(min(pcmod.C_avg_per_epoch),3)
# C_round_max = round(max(pcmod.C_avg_per_epoch),3)

# # E+C
# Eavg_plus_Cavg_per_epoch = pcmod.E_avg_per_epoch + pcmod.C_avg_per_epoch
# EC_first = round_first + C_round_first
# EC_last = round_last + C_round_last
# EC_min = round_min + C_round_min

# # Accuracy
# acc_first = round(pcmod.acc_per_epoch[0],1)
# acc_last = round(pcmod.acc_per_epoch[-1],1)
# acc_max = round(max(pcmod.acc_per_epoch),1)


# # general variables
# num_epochs = range(1, pcmod.p.num_epochs+1)
# representation_cost = pcmod.E_avg_per_epoch
# classification_cost = pcmod.C_avg_per_epoch
# accuracy = pcmod.acc_per_epoch
# class_type = pcmod.class_type
# prior_type = pcmod.p.r_prior


# fig, (axE) = plt.subplots(1)
# fig.suptitle("{}  {}  {}  {}  ".format(pcmod.p.unit_act,prior_type,class_type,pcmod.p.hidden_sizes)+'\n'\
# +'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
# +'Cmax={} '.format(C_round_max)+'Cmin={} '.format(C_round_min) + 'Amax={} '.format(acc_max))


# twinEA = axE.twinx()

# # set colors
# plotE = axE.plot(num_epochs, representation_cost, '#000000', label="Avg E")
# plotC = axE.plot(num_epochs, classification_cost, '#4363d8', label="Avg C")
# plotEA = twinEA.plot(num_epochs, accuracy, 'darkgreen', label="Accuracy")

# # set limits for and label x,y-axes for both subplots

# # axE.set_xlim(0, 2)
# axE.set_ylim(0, 1)
# twinEA.set_ylim(0, 100)

# axE.set_xlabel("Epoch")
# axE.set_ylabel("Avg E, Avg C")
# twinEA.set_ylabel("Accuracy")

# # show plot
# plt.show()



"""
Plotting EVALUATION E, C, and Acc of a Trained or Untrained, Classifying or Non-Classifying Model

DIFFERENT SCALES for E and C

Generates Two Subplots: E & A on the top, and C & A on the bottom
"""

# # load model as a pickle
# prior_type = 'gauss'
# # prior_type = 'kurt'

# # class_type = 'NC'
# # class_type = 'C1'
# class_type = 'C2'


# # load evaluated model and objects containing evaluation metrics
# evaluation_in = open('evaluate_untrainedmodel_on_100x10.pydb','rb')
# pcmod,E,C,Classif_success_by_img,Acc = pickle.load(evaluation_in)
# evaluation_in.close()

# n_eval_images = pcmod.n_eval_images
# n_eval_images = range(1,n_eval_images+1)


# # E,C and Accuracy data points for plotting

# # E

# # # double check that evaluation objects contain the right values
# # print(E)
# # print(C)
# # print(Classif_success_by_img)
# # print(Acc)

# Eavg = round((sum(E)/pcmod.n_eval_images),2)
# Emin = round(min(E),2)

# # C
# Cavg = round((sum(C)/pcmod.n_eval_images),2)
# Cmin = round(min(C),2)


# # Acc
# Acc = Acc



# # general variables
# eval_class_type = pcmod.eval_class_type
# prior_type = pcmod.p.r_prior

# # plot E/Acc vs epoch; plot C/Acc vs epoch

# # split into vertically-stacked subplots

# fig, (axE, axC) = plt.subplots(2)
# fig.suptitle("{}  {}  eval classif type={}     ".format(pcmod.p.unit_act,prior_type,eval_class_type)+'\n'\
# +'Eavg={} '.format(Eavg)+'Emin={} '.format(Emin)+ 'Cavg={} '.format(Cavg) +'Cmin={} '.format(Cmin)+ 'Acc={} '.format(Acc))


# # create labeled plot objects
# # black, sky,
# plotE = axE.plot(n_eval_images, E, '#000000', label="E")
# plotC = axC.plot(n_eval_images, C, '#4363d8', label="C")


# # set limits for and label x,y-axes for both subplots

# # E plotting range is around 2000 if model is untrained; if trained on 100 images, E ~ 10; if trained on 1000 images, E ~ 1.
# axE.set_ylim(0, 2000)
# # C plotting range is around 20 if model is untrained; if trained on 100 images, E ~ 0.15; if trained on 1000 images, E ~ 0.05.
# axC.set_ylim(0, 20)


# axE.set_xlabel("Evaluation Image")
# axE.set_ylabel("E")


# axC.set_xlabel("Evaluation Image")
# axC.set_ylabel("C")


# # show plot
# plt.show()


"""
Plotting EVALUATION E, C, and Acc of a Trained or Untrained, Classifying or Non-Classifying Model

SAME SCALE for E and C

Generates One Plot: E, C & A Together
"""

# # load trained model as a pickle
# prior_type = 'gauss'
# # prior_type = 'kurt'

# # class_type = 'NC'
# # class_type = 'C1'
# class_type = 'C2'


# # load evaluated model and objects containing evaluation metrics
# evaluation_in = open('evaluate_untrainedmodel_on_100x10.pydb','rb')
# pcmod,E,C,Classif_success_by_img,Acc = pickle.load(evaluation_in)
# evaluation_in.close()

# n_eval_images = pcmod.n_eval_images
# n_eval_images = range(1,n_eval_images+1)


# # E,C and Accuracy data points for plotting

# # E

# # # double check that evaluation objects contain the right values
# # print(E)
# # print(C)
# # print(Classif_success_by_img)
# # print(Acc)

# Eavg = round((sum(E)/pcmod.n_eval_images),2)
# Emin = round(min(E),2)

# # C
# Cavg = round((sum(C)/pcmod.n_eval_images),2)
# Cmin = round(min(C),2)

# # Acc
# Acc = Acc


# # general variables
# eval_class_type = pcmod.eval_class_type
# prior_type = pcmod.p.r_prior



# fig, (axE) = plt.subplots(1)
# fig.suptitle("{}  {}  eval_class_type={}  {}  ".format(pcmod.p.unit_act,prior_type,eval_class_type,pcmod.p.hidden_sizes)+'\n'\
# +'Eavg={} '.format(Eavg)+'Emin={} '.format(Emin)\
# +'Cavg={} '.format(Cavg)+'Cmin={} '.format(Cmin) + 'Acc={} '.format(Acc))


# # set colors
# plotE = axE.plot(n_eval_images, E, '#000000', label="E")
# plotC = axE.plot(n_eval_images, C, '#4363d8', label="C")

# # set limits for and label x,y-axes for both subplots

# # E/C plotting range is around 2000 if model is untrained; if trained on 100 images, E/C ~~ 10; if trained on 1000 images, E/C ~~ 1.
# axE.set_ylim(0, 2000)


# axE.set_xlabel("Evaluation Image")
# axE.set_ylabel("E, C")

# # show plot
# plt.show()



"""
Plotting PREDICTION Errors (L1 & L2) of a Trained or Untrained Model

Sequentially generates one PE plot per image predicted
"""


# # import model and predicted image set (could contain multple 2-dim images, or one 3-dim multi-image
# # vector, like X_train)
# prediction_in = open('predict_imgs_1-5_with_trained_1000_imgs_gauss_NC.pydb','rb')
# pcmod, prediction_image_set, n_pred_images = pickle.load(prediction_in)
# prediction_in.close()


# # model variables
# transform_function = pcmod.p.unit_act
# classification_type_during_training = pcmod.class_type
# prior_type = pcmod.p.r_prior
# num_updates = range(1,pcmod.n_pred_updates+1)


# print(pcmod.prediction_errors_l1[0])
# print(pcmod.prediction_errors_l1[0][-1])
# print(pcmod.n_pred_images)


# for prediction_image in range(0,n_pred_images):
    
#     # prediction specific variables
    
#     pe_1_first = round(pcmod.prediction_errors_l1[prediction_image][0],1)
#     pe_1_last = round(pcmod.prediction_errors_l1[prediction_image][-1],1)
#     pe_2_first = round(pcmod.prediction_errors_l2[prediction_image][0],1)
#     pe_2_last = round(pcmod.prediction_errors_l2[prediction_image][-1],1)
    
    
#     fig, ax = plt.subplots(1)
#     fig.suptitle("{}  {}  {}  pred_img {}  ".format(pcmod.p.unit_act,pcmod.class_type,prior_type,prediction_image+1)+'\n'\
#     +'pe_1_first={} '.format(pe_1_first)+'pe_1_last={} '.format(pe_1_last)\
#     + 'pe_2_first={} '.format(pe_2_first) + 'pe_2_last={} '.format(pe_2_last))
    
    
#     # black and navy
#     plotE = ax.plot(num_updates, pcmod.prediction_errors_l1[prediction_image], '#000000', label="pe_1")
#     plotE = ax.plot(num_updates, pcmod.prediction_errors_l2[prediction_image], '#000075', label="pe_2")
#     ax.set_ylim(0, 50)
#     ax.legend()
    
#     ax.set_xlabel("Update")
#     ax.set_ylabel("L1, L2 PE")
    
    
#     plt.show()









"""
Storage
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


# old plot: example of plt.annotate method

# # plot E results same learning rate all layers
# plt.plot(epoch+1, E_avg_per_epoch, '.k')
# plt.title("{}-HL Model".format(pcmod.n_hidden_layers) + '\n' + "k_r = {}".format(k_r) \
# + '\n' + "k_U = {}".format(k_U))
# if epoch == pcmod.p.num_epochs-1:
#     plt.annotate("E avg initial = {}".format(round_first) + '\n' \
#     + "E avg final = {}".format(round_last) + '\n' \
#     + "E avg min = {}".format(round_min) + '\n' \
#     + "E avg total descent = {}".format(round((round_first - round_last),1)), (0.58,0.67), xycoords='figure fraction')

#     plt.xlabel("epoch ({})".format(pcmod.p.num_epochs))
#     plt.ylabel("E avg")
