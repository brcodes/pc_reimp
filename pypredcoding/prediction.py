"""
Script for using trained PredictiveCodingClassifier to generate and return
predicted image(s) from input image(s)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


"""
Pickle In Model to Use for Prediction
MUST comment-in desired naming parameters
"""

# import the right model to predict with
# by uncommenting all of its parameters

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
num_epochs = '100e'
# num_epochs = '50e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
training_dataset = 'tanh100x10'
# training_dataset = 'tanh10x10'
# training_dataset = '-'

#evaluated or not evaluated with so far
# evaluated = 'E'
evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
# eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh10x10'
eval_dataset = '-'

#used or not used for prediction so far
# used_for_pred = 'P'
used_for_pred = 'np'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
# pred_dataset = '5imgs'
pred_dataset = '-'

#extra identifier for any particular or unique qualities of the model object
# extra_tag = 'randUo'
# extra_tag = 'pipeline_test'
extra_tag = '-'


# SOME CONDITIONAL LOGIC:
# If the object has been evaluated before, a different pickle load (of a tuple) is required
# If it hasn'e been evaluated, the pickle load will be a single model object, pcmod

# NOTE: pipeline only works in order: main.py -> evaluate.py -> predict.py -> plotting.py
# or main.py -> evaluate.py -> plotting.py
# or main.py -> plotting.py

if evaluated == "E":
    # load it
    pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
      trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
    pcmod,E,C,Classif_success_by_img,Acc = pickle.load(pcmod_in)
    pcmod_in.close()

elif evaluated == "ne":
    # load it
    pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
      trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
    pcmod = pickle.load(pcmod_in)
    pcmod_in.close()

else:
    print('evaluated must = "E" or "ne"')


"""
Pickle In Prediction Image Set
comment-in correct image set
"""

# NOTE: you must import all of (e.g.) tanh100x10.pydb to get the 5 in-bag-to-lena-prewhitened prediction images,
# even if you don't intend to use X_train or y_train

# linear_data_in = open('linear_10x10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(linear_data_in)
# linear_data_in.close()

# # prediction dataset
# tanh_data_in = open('tanh_10x10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()

# prediction dataset
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

# # prediction dataset
# tanh_data_in = open('tanh_1000x10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()





"""
Set Prediction Images
add desired prediction image vectors to the list "prediction_image_set"
will work with 1-n number of single images (e.g. lena_zoom) or 3-dim image vectors (e.g. X_train)
"""


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


# print("training_img.shape")
# print(training_img.shape)


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

# print("combined pred imgs vector shape")
# print(combined_pred_imgs_vec.shape)


"""
Predict
"""


# predict each image in the multi-image vector created above
pcmod.predict(combined_pred_imgs_vec)


"""
Pickle Out Model Used for Prediction, the Prediction Image Set used, and # Prediction Images 
(all other useful PE metrics for plotting are inside pcmod)

MUST comment-in (or write new) name of image set predicted
do not touch used_for_pred = 'P'
"""

#output pickle naming

#used for prediction is now true
used_for_pred = 'P'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
pred_dataset = '5imgs'



# pickle the model (contains self.variables for prediction plotting)
prediction_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
pickle.dump((pcmod, prediction_image_set, n_pred_images), prediction_out)
prediction_out.close()
