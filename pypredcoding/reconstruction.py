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
model_size = '[36.32]'
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
# num_epochs = '100e'
num_epochs = '40e'
# num_epochs = '50e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
# training_dataset = 'tanh100x10'
training_dataset = 'tanh100x10_size_24x24'
# training_dataset = 'tanh10x10'
# training_dataset = '-'

#evaluated or not evaluated with so far
evaluated = 'E'
# evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
# eval_dataset = 'tanh100x10'
eval_dataset = 'tanh100x10_size_24x24'
# eval_dataset = 'tanh10x10'
# eval_dataset = '-'

#must be P
used_for_pred = 'P'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
# pred_dataset = '5imgs'
# pred_dataset = '0-9_minE_128.32_kurt'
pred_dataset = '0-9_minE_36.32'
# pred_dataset = '0-9_maxE_128.32_kurt'
# pred_dataset = '-'

#extra identifier for any particular or unique qualities of the model object
# extra_tag = 'randUo'
# extra_tag = 'pipeline_test'
extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
# extra_tag = '-'


# import model and predicted image set (set could contain multple 2-dim images, or one 3-dim multi-image
# vector, like X_train)
prediction_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod, prediction_image_set, n_pred_images = pickle.load(prediction_in)
prediction_in.close()

# for loading and recontruction of a different prediction_image_set
other_prediction_in = open('pc.[36.32].tanh.gauss.NC.T.40e.tanh100x10.E.tanh100x10.P.0-9_minE_36.32.-.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
other_pcmod, other_prediction_image_set, other_n_pred_images = pickle.load(other_prediction_in)
other_prediction_in.close()


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
#     # # original image from other source
#     # original_image = other_prediction_image_set[image].reshape(28,28)
    
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
    
    
# TILED VERSION BELOW
# !!!!

for image in range(0,n_pred_images):
    
    # layer 1 reconstruction
    num_modules = pcmod.U[1].shape[0]
    print('num modules is {}'.format(num_modules))
    
    rthird = int(pcmod.r[1].shape[0]/3)
    print('rthird is {}'.format(rthird))
    rindex1 = 0
    rindex2 = rthird
    
    for mod in range(0,num_modules):
        print('len(pcmod.r1s) is {}'.format(len(pcmod.r1s)))
        print('shape U[1] is {}'.format(pcmod.U[1].shape))
        
        
        
        fU1r1_l1 = tanh_trans(pcmod.U[1][mod].dot(pcmod.r1s[image][rindex1:rindex2]))[0]
        #test number of r[1]s is correct (should = num prediction images)
        
        fU1r1_resize_l1 = fU1r1_l1.reshape(24,12)
        
        # layer 2 reconstruction
        fU2r2 = tanh_trans(pcmod.U[2].dot(pcmod.prediction[image]))[0]
        # tests
        # print("shape of U2 {} and r2 {}".format(pcmod.prediction[image].shape, pcmod.U[2].shape))
        # print("shape of fU2r2 {}".format(fU2r2.shape))
        # print("shape of U1 {}".format(pcmod.U[1]))
        fU1r1_l2 = tanh_trans(pcmod.U[1][mod].dot(fU2r2[rindex1:rindex2]))[0]
        fU1r1_resize_l2 = fU1r1_l2.reshape(24,12)
        
        # original image
        # original_image = prediction_image_set[image].reshape(28,28)
        # original image from other source
        original_image = prediction_image_set[image].reshape(24,24)
        
        # plot
        plt.subplot(131),plt.imshow(original_image, cmap='Greys'),plt.title('image #{} Original'.format(image+1))
        plt.xticks([]), plt.yticks([])
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(132),plt.imshow(fU1r1_resize_l1, cmap='Greys'),plt.title('image #{} L{} tile{}'.format(image+1,1,mod+1))
        plt.xticks([]), plt.yticks([])
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(133),plt.imshow(fU1r1_resize_l2, cmap='Greys'),plt.title('image #{} L{} tile{}'.format(image+1,2,mod+1))
        plt.xticks([]), plt.yticks([])
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.show()
        
        rindex1 += rthird
        rindex2 += rthird

    
"""
Pickle In Model to Plot Its U1 Basis Vectors (Usually a Trained Model)
MUST comment-in desired naming parameters
"""


#model size
# model_size = '[32.10]'
# model_size = '[32.32]'
model_size = '[36.32]'
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
# num_epochs = '100e'
num_epochs = '40e'
# num_epochs = '50e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
# training_dataset = 'tanh100x10'
training_dataset = 'tanh100x10_size_24x24'
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
extra_tag = 'tile_offset_6_poly_lr_0.05_lU_0.005_me40_pp1'
# extra_tag = '-'

# load it
pcmod_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod = pickle.load(pcmod_in)
pcmod_in.close()



"""
Extract and Plot U[1] Basis Vectors
"""

# # NON TILED VERSION BELOW

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
    


# print('shape of r0 is {}'.format(pcmod.r[0].shape))
# print('shape of r1 is {}'.format(pcmod.r[1].shape))
# print('shape of U1 is {}'.format(pcmod.U[1].shape))
# print('shape of r2 is {}'.format(pcmod.r[2].shape))
# print('shape of U2 is {}'.format(pcmod.U[2].shape))



# TILED VERSION BELOW
# !!!!


# U1 = np.concatenate((pcmod.U[1][0],pcmod.U[1][1],pcmod.U[1][2]),axis=0)

# print("lenU1")
# print(len(U1))
# print('shapeU1')
# print(U1.shape)


# num_basis_vecs = U1.shape[1]

# # average the basis vector values created by overlapping tiles

# q1 = U1[0:144,:]
# q2 = (U1[144:288,:]+U1[288:432,:])/2
# q3 = (U1[432:576,:]+U1[576:720,:])/2
# q4 = U1[720:864,:]

# U1_trunc = np.concatenate((q1,q2,q3,q4),axis=0)

# print('shape of U1_trunc is {}'.format(U1_trunc.shape))


# for i in range(0, num_basis_vecs):
    
#     print('basis #{}'.format(i+1))
    
#     basis_1d = U1_trunc[:,i]
#     print('shape1d basis is {}'.format(basis_1d.shape))

#     basis_2d = basis_1d.reshape(24,24)
#     print('shape2d basis is {}'.format(basis_2d.shape))
    
#     plt.imshow(basis_2d, cmap='Greys'),plt.title('32,32 Gaussian U[1] basis vector #{}'.format(i+1))
#     plt.show()
    


# print('shape of r0 is {}'.format(pcmod.r[0].shape))
# print('shape of r1 is {}'.format(pcmod.r[1].shape))
# print('shape of U1 is {}'.format(pcmod.U[1].shape))
# print('shape of r2 is {}'.format(pcmod.r[2].shape))
# print('shape of U2 is {}'.format(pcmod.U[2].shape))




