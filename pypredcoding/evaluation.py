"""
Script for evaluating PredictiveCodingClassifier model classification accuracy,
representation costs and classification costs against any input image(s)
"""

import pickle
import data

"""
Pickle In
"""

# import the right model to evaluate
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
# num_epochs = '100e'
num_epochs = '50e'
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
extra_tag = 'pipeline_test'
# extra_tag = '-'

# load it
pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

# load data to evaluate against
tanh_data_in = open('tanh_10x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

# print(X_train.shape)
# print(y_train.shape)

# Unflatten
X_inflated = data.inflate_vectors(X_train)

# print(X_inflated.shape)

"""
Evaluate
"""


# evaluate
E,C,Classif_success_by_img,Acc = pcmod.evaluate(X_inflated,y_train)


"""
Pickle Out
"""

#output pickle naming

#evaluated is now true
evaluated = 'E'

#images evaluated against
# eval_dataset = 'tanh100x10'
eval_dataset = 'tanh10x10'


evaluation_out = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
    trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'wb')
pickle.dump((pcmod,E,C,Classif_success_by_img,Acc), evaluation_out)
evaluation_out.close()
