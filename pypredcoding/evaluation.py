"""
Script for evaluating PredictiveCodingClassifier model classification accuracy,
representation costs and classification costs against any input image(s)
"""

import pickle
import data

# load trained/untrained model object to evaluate
pcmod_in = open('pcmod_untrained.pydb','rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()

# load data to evaluate against
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

# print(X_train.shape)
# print(y_train.shape)

X_inflated = data.inflate_vectors(X_train)

# print(X_inflated.shape)


# evaluate
E,C,Classif_success_by_img,Acc = pcmod.evaluate(X_inflated,y_train)


#NOTE: Change the pickle name by hand for now!!!

# name format: evaluation_of_<type of model evaluated>_on_<data evaluated against>.pydb
evaluation_out = open('evaluate_untrainedmodel_on_100x10.pydb','wb')
pickle.dump((pcmod,E,C,Classif_success_by_img,Acc), evaluation_out)
evaluation_out.close()
