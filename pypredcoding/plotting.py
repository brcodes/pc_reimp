"""
Script for plotting data from trained PredictiveCodingClassifier object
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# load trained model as a pickle

pcmod_in = open('pcmod_trained.pydb','rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

# display total number of model parameters after training

print('Total number of model parameters')
print(pcmod.n_model_parameters)
print('\n')

# E,C and Accuracty data points for plotting

# E
round_first = round(pcmod.E_avg_per_epoch[0],1)
round_epoch1 = round(pcmod.E_avg_per_epoch[1],1)
round_last = round(pcmod.E_avg_per_epoch[-1],1)
round_min = round(min(pcmod.E_avg_per_epoch),1)

# C
C_round_first = round(pcmod.C_avg_per_epoch[0],1)
C_round_last = round(pcmod.C_avg_per_epoch[-1],1)
C_round_min = round(min(pcmod.C_avg_per_epoch),1)
C_round_max = round(max(pcmod.C_avg_per_epoch),1)

# E+C
Eavg_plus_Cavg_per_epoch = pcmod.E_avg_per_epoch + pcmod.C_avg_per_epoch
EC_first = round_first + C_round_first
EC_last = round_last + C_round_last
EC_min = round_min + C_round_min

# Accuracy
acc_first = round(pcmod.acc_per_epoch[0],1)
acc_last = round(pcmod.acc_per_epoch[-1],1)
acc_max = round(max(pcmod.acc_per_epoch),1)


# general variables
num_epochs = range(1, pcmod.p.num_epochs+1)
representation_cost = pcmod.E_avg_per_epoch
classification_cost = pcmod.C_avg_per_epoch
accuracy = pcmod.acc_per_epoch
class_type = pcmod.class_type



"""
Classification Data Plotting
"""

# plot E/Acc vs epoch; plot C/Acc vs epoch

# split into vertically-stacked subplots

fig, (axE, axC) = plt.subplots(2)
fig.suptitle("{}  {}      ".format(pcmod.p.unit_act,class_type)+"lr_r={} ".format(pcmod.lr_r)+"lr_U={} ".format(pcmod.lr_U)+"lr_o={}".format(pcmod.lr_o)+'\n'\
+'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
+'Cmax={} '.format(C_round_max)+'Cmin={} '.format(C_round_min) + 'Amax={} '.format(acc_max))

# create a second y-axis (Accuracy) for each subplot

twinEA = axE.twinx()
twinCA = axC.twinx()

# create labeled plot objects

plotE = axE.plot(num_epochs, representation_cost, "r-", label="Avg E")
plotC = axC.plot(num_epochs, classification_cost, "b-", label="Avg C")
plotEA = twinEA.plot(num_epochs, accuracy, "g-", label="Accuracy")
plotCA = twinCA.plot(num_epochs, accuracy, "g-", label="Accuracy")

# set limits for and label x,y-axes for both subplots

# axE.set_xlim(0, 2)
axC.set_ylim(0, 0.3)
twinEA.set_ylim(0, 100)
twinCA.set_ylim(0, 100)

axE.set_xlabel("Epoch")
axE.set_ylabel("Avg E")
twinEA.set_ylabel("Accuracy")

axC.set_xlabel("Epoch")
axC.set_ylabel("Avg C")
twinCA.set_ylabel("Accuracy")

# pass labeled plot objects legend method

axE.legend(handles=[plotE, plotEA])
axC.legend(handles=[plotC, plotCA])

# show plot

plt.show()



"""
Plotting When No Classification (e.g. Training Model for Prediction)
"""



# fig, ax = plt.subplots(1)
# fig.suptitle("{}  {}      ".format(pcmod.p.unit_act,class_type)+"lr_r={} ".format(pcmod.lr_r)+"lr_U={} ".format(pcmod.lr_U)+"lr_o={}".format(pcmod.lr_o)+'\n'\
# +'E1={} '.format(round_epoch1)+'Emin={} '.format(round_min)\
# + 'Amax={} '.format(acc_max))

# twinA = ax.twinx()


# plotE = ax.plot(num_epochs, representation_cost, "r-", label="Avg E")
# plotA = twinA.plot(num_epochs, accuracy, "g-", label="Accuracy")

# twinA.set_ylim(0, 100)

# ax.set_xlabel("Epoch")
# ax.set_ylabel("Avg E")
# twinA.set_ylabel("Accuracy")


# plt.show()




















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


# # plot C results same learning rate all layers
# plt.plot(epoch+1, C_avg_per_epoch, '.k')
# plt.title("{}-HL {} Model".format(pcmod.n_hidden_layers,class_type) + '\n' + "k_r = {}".format(k_r) \
# + '\n' + "k_U = {}".format(k_U))
# if epoch == pcmod.p.num_epochs-1:
#     plt.annotate("C avg initial = {}".format(C_round_first) + '\n' \
#     + "C avg final = {}".format(C_round_last) + '\n' \
#     + "C avg min = {}".format(C_round_min) + '\n' \
#     + "C avg total descent = {}".format(round((C_round_first - C_round_last),1)), (0.58,0.67), xycoords='figure fraction')

#     plt.xlabel("epoch ({})".format(pcmod.p.num_epochs))
#     plt.ylabel("C avg")


# # plot E + C results same learning rate all layers
# plt.plot(epoch+1, Eavg_plus_Cavg_per_epoch, '.k')
# plt.title("{}-HL {} Model".format(pcmod.n_hidden_layers,class_type) + '\n' + "k_r = {}".format(k_r) \
# + '\n' + "k_U = {}".format(k_U))
# if epoch == pcmod.p.num_epochs-1:
#     plt.annotate("E+C initial = {}".format(EC_first) + '\n' \
#     + "E+C final = {}".format(EC_last) + '\n' \
#     + "E+C min = {}".format(EC_min) + '\n' \
#     + "E+C total descent = {}".format(round((EC_first - EC_last),1)), (0.58,0.67), xycoords='figure fraction')

#     plt.xlabel("epoch ({})".format(pcmod.p.num_epochs))
#     plt.ylabel("E+C")

# # plot Accuracy results same learning rate all layers
# plt.plot(epoch+1, acc_per_epoch, '.k')
# plt.title("{}-HL {} Model".format(pcmod.n_hidden_layers,class_type) + '\n' + "k_r = {}".format(k_r) \
# + '\n' + "k_U = {}".format(k_U))
# if epoch == pcmod.p.num_epochs-1:
#     plt.annotate("Accuracy initial = {}".format(acc_first) + '\n' \
#     + "Accuracy final = {}".format(acc_last) + '\n' \
#     + "Accuracy max = {}".format(acc_max) + '\n' \
#     + "Accuracy total ascent = {}".format(round((acc_last - acc_first),1)), (0.58,0.67), xycoords='figure fraction')

#     plt.xlabel("epoch ({})".format(pcmod.p.num_epochs))
#     plt.ylabel("Accuracy")


# # plot E results of asymmetry experiments, i.e. different learning rates for i layers and n layer
# plt.plot(epoch+1, E_avg_per_epoch, '.k')
# plt.title("{}-HL Model".format(pcmod.n_hidden_layers) + '\n' + "L1 k_(r,U) = {}".format(L1_k) \
# + '\n' + "L2 k_(r,U) = {}".format(L2_k))
# if epoch == pcmod.p.num_epochs-1:
#     plt.annotate("E avg initial = {}".format(round_first) + '\n' \
#     + "E avg final = {}".format(round_last) + '\n' \
#     + "E avg min = {}".format(round_min) + '\n' \
#     + "E avg total descent = {}".format(round((round_first - round_last),1)), (0.58,0.67), xycoords='figure fraction')
#     plt.xlabel("epoch ({})".format(pcmod.p.num_epochs))
#     plt.ylabel("E avg")

# print("Average representation error per each epoch ({} total), in format [E_epoch1, E_epoch2...]".format(pcmod.p.num_epochs))
# print(pcmod.E_avg_per_epoch)
# print('\n')
# print("Average classification error per each epoch ({} total), in format [C_epoch1, C_epoch2...]".format(pcmod.p.num_epochs))
# print(pcmod.C_avg_per_epoch)
# print('\n')
# print("Model trained.")
# print('\n')
