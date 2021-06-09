"""
Script for 1) reconstructing predicted input images using a 2-layered PC model
and for 2) plotting U[1] basis vectors of a model of any size
"""

"""
Transformation Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# activation functions
def linear_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Linear transformation. """
    f = U_dot_r
    F = np.eye(len(f))
    return (f, F)



def tanh_trans(U_dot_r):
    """ Though intended to operate on some U.dot(r), will take any numerical
    argument x and return the tuple (f(x), F(x)). Tanh transformation. """
    f = np.tanh(U_dot_r)
    F = np.diag(1 - f.flatten()**2)
    return (f, F)


"""
Pickle In Model That Has Already Been Used to Predict Images of Interest
MUST comment-in desired naming parameters
"""

# import the right model to predict with
# by uncommenting all of its parameters

#model size
# model_size = '[32.10]'
# model_size = '[32.32]'
model_size = '[128.32]'


#transformation function
transform_type = 'tanh'
# transform_type = 'linear'

#prior type
# prior_type = 'gauss'
prior_type = 'kurt'

#classification method
class_type = 'NC'
# class_type = 'C1'
# class_type = 'C2'

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
evaluated = 'E'
# evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh10x10'
# eval_dataset = '-'

#must be P
used_for_pred = 'P'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
# pred_dataset = '5imgs'
# pred_dataset = '0-9_minE_128.32_kurt'
pred_dataset = '0-9_maxE_128.32_kurt'
# pred_dataset = '-'

#extra identifier for any particular or unique qualities of the model object
# extra_tag = 'randUo'
# extra_tag = 'pipeline_test'
extra_tag = '-'


# # import model and predicted image set (set could contain multple 2-dim images, or one 3-dim multi-image
# # vector, like X_train)
# prediction_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
#   trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
# pcmod, prediction_image_set, n_pred_images = pickle.load(prediction_in)
# prediction_in.close()


"""
Reconstruct and Plot Prediction Images
"""

# for image in range(0,n_pred_images):
    
#     # layer 1 reconstruction
#     fU1r1_l1 = tanh_trans(pcmod.U[1].dot(pcmod.r1s[image]))[0]
#     #test number of r[1]s is correct (should = num prediction images)
#     #print(len(pcmod.r1s))
#     fU1r1_resize_l1 = fU1r1_l1.reshape(28,28)
    
#     # layer 2 reconstruction
#     fU2r2 = tanh_trans(pcmod.U[2].dot(pcmod.prediction[image]))[0]
#     # tests
#     # print("shape of U2 {} and r2 {}".format(pcmod.prediction[image].shape, pcmod.U[2].shape))
#     # print("shape of fU2r2 {}".format(fU2r2.shape))
#     # print("shape of U1 {}".format(pcmod.U[1]))
#     fU1r1_l2 = tanh_trans(pcmod.U[1].dot(fU2r2))[0]
#     fU1r1_resize_l2 = fU1r1_l2.reshape(28,28)
    
#     # original image
#     original_image = prediction_image_set[image].reshape(28,28)
    
#     # plot
#     plt.subplot(131),plt.imshow(original_image, cmap='Greys'),plt.title('image #{} Original'.format(image+1))
#     plt.xticks([]), plt.yticks([])
#     # plt.colorbar(fraction=0.046, pad=0.04)
#     plt.subplot(132),plt.imshow(fU1r1_resize_l1, cmap='Greys'),plt.title('image #{} L{}'.format(image+1,1))
#     plt.xticks([]), plt.yticks([])
#     # plt.colorbar(fraction=0.046, pad=0.04)
#     plt.subplot(133),plt.imshow(fU1r1_resize_l2, cmap='Greys'),plt.title('image #{} L{}'.format(image+1,2))
#     plt.xticks([]), plt.yticks([])
#     # plt.colorbar(fraction=0.046, pad=0.04)
#     plt.show()
    
    
"""
Pickle In Model to Plot Its U1 Basis Vectors (Usually a Trained Model)
MUST comment-in desired naming parameters
"""


#model size
# model_size = '[32.10]'
model_size = '[32.32]'
# model_size = '[128.32]'


#transformation function
transform_type = 'tanh'
# transform_type = 'linear'

#prior type
prior_type = 'gauss'
# prior_type = 'kurt'

#classification method
class_type = 'NC'
# class_type = 'C1'
# class_type = 'C2'

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

# load it
pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()



"""
Extract and Plot U[1] Basis Vectors
"""

# U1 = pcmod.U[1]


# print("lenU1")
# print(len(U1))
# print('shapeU1')
# print(U1.shape)

# num_basis_vecs = U1.shape[1]


# for i in range(0, num_basis_vecs):
#     print('basis #{}'.format(i+1))
    
#     basis_1d = U1[:,i]
#     print('shape1d basis is {}'.format(basis_1d.shape))
    
#     basis_2d = basis_1d.reshape(28,28)
#     print('shape2d basis is {}'.format(basis_2d.shape))
    
#     plt.imshow(basis_2d, cmap='Greys'),plt.title('32,32 Gaussian U[1] basis vector #{}'.format(i+1))
#     plt.show()
    


print('shape of r0 is {}'.format(pcmod.r[0].shape))
print('shape of r1 is {}'.format(pcmod.r[1].shape))
print('shape of U1 is {}'.format(pcmod.U[1].shape))
print('shape of r2 is {}'.format(pcmod.r[2].shape))
print('shape of U2 is {}'.format(pcmod.U[2].shape))






