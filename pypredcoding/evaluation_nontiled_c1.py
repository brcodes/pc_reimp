"""
Script for evaluating PredictiveCodingClassifier model classification accuracy,
representation costs and classification costs against any input image(s)
"""

import pickle
import data

"""
Pickle In Trained or Untrained Model
MUST comment-in desired naming parameters
"""

# import the right model to evaluate
# by uncommenting all of its parameters

#model size
# model_size = '[36.10]'
# model_size = '[288.10]'
# model_size = '[2304.10]'
model_size = '[18432.10]'


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

#trained or untrained
trained = 'T'
# trained = 'nt'

#number of epochs if trained (if not, use -)
# num_epochs = '1000e'
# num_epochs = '100e'
# num_epochs = '40e'
# num_epochs = '50e'
# num_epochs = '20e'
num_epochs = '10e'
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
# extra_tag = 'pipeline_test
# extra_tag = 'tile_offset_6'
# extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
# extra_tag = 'tile_offset_8'
# extra_tag = 'tile_offset_0'
# extra_tag = 'tile_offset_6_poly_lr_0.005_lU_0.005_me40_pp1'
# extra_tag = 'cboost_1'
extra_tag = '-'

# load it
pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

# load data to evaluate against
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

# # load data to evaluate against
# tanh_data_in = open('tanh_100x10_fashion_mnist.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()

# # load data to evaluate against
# tanh_data_in = open('tanh_100x10_cifar10.pydb','rb')
# X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
# tanh_data_in.close()


#output pickle naming

#images evaluated against (must match tanh_data_in]
# eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh100x10_fashion_mnist'
eval_dataset = 'tanh100x10_cifar10'
# eval_dataset = 'tanh10x10'

# print(X_train.shape)
# print(y_train.shape)

# Unflatten
X_inflated = data.inflate_vectors(X_train)

# print(X_inflated.shape)

"""
Evaluate
"""

# list naming parameters above: if anything left unset, evaluate) will not run
naming_parameters = [model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag]


# evaluate
E,C,Classif_success_by_img,Acc = pcmod.evaluate(X_inflated,y_train)


"""
Pickle Out Evaluated Model and E,C,Accuracy Evaluation Metrics
MUST comment-in name of image set evaluated against
do not touch evaluated = 'E'
"""

#output pickle naming

#evaluated is now true
evaluated = 'E'


evaluation_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
pickle.dump((pcmod,E,C,Classif_success_by_img,Acc), evaluation_out)
evaluation_out.close()
