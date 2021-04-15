"""
Script for using trained PredictiveCodingClassifier to generate and return
predicted image(s) from input image(s)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# class_type = 'NC'
# class_type = 'C1'
class_type = 'C2'


pcmod_in = open('pcmod_trained_{}.pydb'.format(class_type),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

tanh_data_in = open('tanh_10x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()


# pcmod.predict(training_img,class_type,"training image", 500)

# pcmod.predict(non_training_img,class_type,"non training mnist image", 500)

pcmod.predict(scrm_training_img,class_type,"scrambled training image", 500)

