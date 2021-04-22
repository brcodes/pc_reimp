"""
Script for using trained PredictiveCodingClassifier to generate and return
predicted image(s) from input image(s)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

prior_type = 'gauss'
# prior_type = 'kurt'

class_type = 'NC'
# class_type = 'C1'
# class_type = 'C2'


pcmod_in = open('pcmod_trained_{}_{}.pydb'.format(prior_type,class_type),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

tanh_data_in = open('tanh_10x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()


# predict argument format
# predict(self,X,X_name,prior_type,classif_type,num_updates)

# pcmod.predict(training_img,"training image",prior_type,class_type, 500)

# pcmod.predict(non_training_img,"non training mnist image",prior_type,class_type, 500)

# pcmod.predict(scrm_training_img,"scrambled training image",prior_type,class_type, 500)

pcmod.predict(lena_pw,"lena prewhitened",prior_type,class_type, 500)

# pcmod.predict(lena_zoom,"lena zoom",prior_type,class_type, 500)
