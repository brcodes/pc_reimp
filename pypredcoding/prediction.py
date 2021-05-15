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


pcmod_in = open('pcmod_trained_1000imgs_{}_{}.pydb'.format(prior_type,class_type),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()


# predict only accepts single [:,:] images or 
# a 3-dim [n,:,:] multi-image vector
# until I have time to do something more clever, I will clump all prediction images
# into [n,:,:] format here


# You must manually change prediction_image_set based on what images you want to predict
# plotting.py will plot prediction error plots in the order of this list
# e.g. if prediction_image_set = [training_img,non_training_img,scrm_training_img,lena_zoom,lena_pw]
# then the plot with "pred_img 1" in the title is the plot corresponding to training_img
# if prediction_image_set = [X_train] from the 1000 image dataset
# then the plot with "pred_img 1" in the title is the plot corresponding to X_train[0]
prediction_image_set = [training_img,non_training_img,scrm_training_img,lena_zoom,lena_pw]

print("training_img.shape")
print(training_img.shape)

# instantiate empty anchor vector to stack on
combined_pred_imgs_vec = np.zeros(shape=(1,28,28))

# count 
n_pred_images = 0

# stack
for image in prediction_image_set:
    if len(image.shape) == 2:
        reshaped = image.reshape(1,28,28)
        combined_pred_imgs_vec = np.vstack((combined_pred_imgs_vec, reshaped))
        n_pred_images += 1
    elif len(image.shape) == 3:
        n_pred_images += image.shape[0]
    else:
        pass
    
# print("combined pred imgs vector shape")
# print(combined_pred_imgs_vec.shape)


# shave off the first empty "anchor" row
combined_pred_imgs_vec = combined_pred_imgs_vec[1:,:,:]

print("combined pred imgs vector shape")
print(combined_pred_imgs_vec.shape)

# predict each image in the multi-image vector created above
pcmod.predict(combined_pred_imgs_vec)

# pickle the model (contains self.variables for prediction plotting)
# NOTE: change most of the name by hand until I fix
prediction_out = open('predict_imgs_1-5_with_trained_1000_imgs_{}_{}.pydb'.format(prior_type, class_type),'wb')
pickle.dump((pcmod, prediction_image_set, n_pred_images), prediction_out)
prediction_out.close()
