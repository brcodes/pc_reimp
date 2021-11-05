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
# model_size = '[32.10]'
# model_size = '[32.32]'
# model_size = '[36.32]'
# model_size = '[128.36]'
model_size = '[512.5]'
# model_size = '[36.36]'
# model_size = '[5.5]'

# transformation function
transform_type = 'tanh'
# transform_type = 'linear'

#prior type
prior_type = 'gauss'
# prior_type = 'kurt'

#classification method
# class_type = 'NC'
class_type = 'C1'
# class_type = 'C2'

#trained or untrained
trained = 'T'
# trained = 'nt'

#number of epochs if trained (if not, use -)
# num_epochs = '5000e'
num_epochs = '1000e'
# num_epochs = '100e'
# num_epochs = '50e'
# num_epochs = '40e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
# training_dataset = 'tanh100x10'
# training_dataset = 'tanh100x10_size_24x24'
# training_dataset = 'rao_ballard_nature_no_pre'
training_dataset = 'rao_ballard_nature'
# training_dataset = 'rao_ballard_nature_dog'
# training_dataset = 'tanh10x10'
# training_dataset = '-'

#evaluated or not evaluated with so far
# evaluated = 'E'
evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
# eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh100x10_size_24x24'
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
# extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
extra_tag = 'C1_LSQ'
# extra_tag = '-'


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

# # prediction dataset
# tanh_data_in = open('tanh_100x10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()

# # prediction dataset
# tanh_tile_data_in = open('tanh_100x10_size_24x24.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_tile_data_in)
# tanh_tile_data_in.close()

# tanh_data_in = open('rao_ballard_nature_no_pre.pydb','rb')
# X_train, y_train = pickle.load(tanh_data_in)
# tanh_data_in.close()

tanh_data_in = open('rao_ballard_nature.pydb','rb')
X_train, y_train = pickle.load(tanh_data_in)
tanh_data_in.close()

# tanh_data_in = open('rao_ballard_nature_dog.pydb','rb')
# X_train, y_train = pickle.load(tanh_data_in)
# tanh_data_in.close()

# # prediction dataset
# tanh_data_in = open('tanh_1000x10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()

"""
Isolate best-performing digit of each kind (0-9)
from evaluation object E (i.e. list pcmod.E_per_image)
(model must have been evaluated on a 100x10 image set for indexing to work)
(remember X_train above is flat)
"""

# E_per_image = E
# E_by_dig = {}
# E_min_per_dig = {}
# E_min_index_per_dig = {}
# lowest_E_digits = []

# index1 = -100
# index2 = -1

# for dig in range(0,10):
#     index1 += 100
#     index2 += 100
#     E_by_dig[dig] = E_per_image[index1:index2]
#     E_min_per_dig[dig] = min(E_by_dig[dig])
#     E_min_index_per_dig[dig] = E_per_image.index(E_min_per_dig[dig])
#     lowest_E_digits.append(X_train[E_min_index_per_dig[dig],:])
#     # add another dimension to make all prediction images unified in 784,1 shape
#     lowest_E_digits[dig] = lowest_E_digits[dig][None,:]

# print("E_min_per_dig")
# print(E_min_per_dig)
# print("E_min_index_per_dig")
# print(E_min_index_per_dig)
    

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


prediction_image_set = X_train

# print("len(prediction_image_set)")
# print(len(prediction_image_set))
# print("prediction_image_set[0].shape")
# print(prediction_image_set[0].shape)

# prediction_image_set = [training_img,non_training_img,scrm_training_img,lena_zoom,lena_pw]

# # print("training_img.shape")
# # print(training_img.shape)


# # instantiate empty anchor vector to stack on
# # # for non tiled
# # combined_pred_imgs_vec = np.zeros(shape=(1,24,24))
# #for tiled
# combined_pred_imgs_vec = np.zeros(shape=(1,28,28))

# # count
# n_pred_images = 0

# # vstack and count number of prediction images

# for image in prediction_image_set:
#     if len(image.shape) == 2:
#         # #for non-tiled
#         # reshaped = image.reshape(1,28,28)
#         #for tiled
#         reshaped = image.reshape(1,28,28)
#         combined_pred_imgs_vec = np.vstack((combined_pred_imgs_vec, reshaped))
#         n_pred_images += 1
#     elif len(image.shape) == 3:
#         n_pred_images += image.shape[0]
#     else:
#         pass

# # print("combined pred imgs vector shape")
# # print(combined_pred_imgs_vec.shape)

# # shave off the first empty "anchor" row
# combined_pred_imgs_vec = combined_pred_imgs_vec[1:,:,:]

# # print("combined pred imgs vector shape")
# # print(combined_pred_imgs_vec.shape)


#If RB Nature imported: Xtrain is fully formed already 
combined_pred_imgs_vec = X_train.reshape((5,28,28))
print('shape of combined pred imgs vec is {}'.format(combined_pred_imgs_vec.shape))


"""
Predict
"""


# list naming parameters above: if anything left unset, predict() will not run
naming_parameters = [model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag]

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
# pred_dataset = '5imgs'
# pred_dataset = 'rao_ballard_nature_no_pre'
pred_dataset = 'rao_ballard_nature'
# pred_dataset = 'rao_ballard_nature_dog'
# pred_dataset = '0-9_minE_128.32_kurt'
# pred_dataset = '0-9_minE_36.32'
# pred_dataset = '0-9_maxE_128.32_kurt'



# pickle the model (contains self.variables for prediction plotting)
prediction_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
pickle.dump((pcmod, prediction_image_set), prediction_out)
prediction_out.close()
