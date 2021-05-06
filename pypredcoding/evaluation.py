"""
Script for evaluating PredictiveCodingClassifier model classification accuracy,
representation costs and classification costs against any input image(s)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# load model to evaluate
pcmod_in = open('pcmod_untrained.pydb','rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

# load data to evaluate against
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

# evaluate
E,C,Classif_success_by_img,Acc = pcmod.evaluate(X_train,y_train)


# name format: evaluation_of_<type of model evaluated>_on_<data evaluated against>.pydb

evaluation_out = open('evaluation_of_untrained_on_100x10.pydb','wb')
pickle.dump((E,C,Classif_success_by_img,Acc), evaluation_out)
evaluation_out.close()
