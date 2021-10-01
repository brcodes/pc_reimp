'''
classification heat maps
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np

#Import evaluated model

#model size
# model_size = '[32.10]'
# model_size = '[32.32]'
# model_size = '[128.32]'
model_size = '[288.288]'
# model_size = '[288.10]'
# model_size = '[96.32]'


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
# num_epochs = '40e'
# num_epochs = '50e'
# num_epochs = '20e'
num_epochs = '10e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
training_dataset = 'tanh100x10'
# training_dataset = 'tanh100x10_size_24x24'
# training_dataset = 'tanh10x10'
# training_dataset = '-'

#evaluated or not evaluated with so far
evaluated = 'E'

#images evaluated against, if evaluated (if not, use -)
eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh10x10'
# eval_dataset = 'ten_of_each_dig_from_mnist_1000'
# eval_dataset = '-'

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
evaluation_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod,E,C,Classif_success_by_img,Acc,softmax_guess_each_img = pickle.load(evaluation_in)
evaluation_in.close()

# # load data that was evaluated against
# tanh_data_in = open('ten_of_each_dig_from_mnist_1000.pydb','rb')
# X_train, y_train = pickle.load(tanh_data_in)
# tanh_data_in.close()

# load data to evaluate against
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()

print('initial shape of X_train is {}'.format(X_train.shape))
print('initial shape of y_train is {}'.format(y_train.shape))

#Fill labels matrix and softmaxed guesses matrix
labels_heatmap_matrix = np.zeros((1,10))
softmax_heatmap_matrix = np.zeros((1,10))

print('size initial heatmap matrix is {}'.format(labels_heatmap_matrix.shape))
print('size initial softmax heatmap matrix is {}'.format(softmax_heatmap_matrix.shape))

dig_index1 = 0
dig_index2 = 100

for digit in range(0,10):
    label_list = np.array(y_train[dig_index1:dig_index2])
    print('label list shape dig {} is {}'.format(digit,label_list.shape))
    labels_heatmap_matrix = np.vstack((labels_heatmap_matrix,label_list))
    softmax_guess_list = np.squeeze(np.array(softmax_guess_each_img[dig_index1:dig_index2]))
    print('softmax guess list shape dig {} is {}'.format(digit,np.array(softmax_guess_list).shape))
    softmax_heatmap_matrix = np.vstack((softmax_heatmap_matrix,softmax_guess_list))
    # print('labels heatmap matrix digit {} is {}'.format(digit,labels_heatmap_matrix))
    # print('softmax heatmap matrix digit {} is {}'.format(digit,softmax_heatmap_matrix))
    dig_index1 += 100
    dig_index2 += 100
    
softmax_heatmap_matrix = softmax_heatmap_matrix[1:,:]
labels_heatmap_matrix = labels_heatmap_matrix[1:,:]
    
print('final labels heatmap matrix shape is {}'.format(np.array(labels_heatmap_matrix).shape))
print('final softmax heatmap matrix shape is {}'.format(np.array(softmax_heatmap_matrix).shape))


# #Labels first
# fig, ax = plt.subplots()
# im = ax.imshow(labels_heatmap_matrix)

# # We want to show all ticks...
# ax.set_xticks(np.arange(10))
# ax.set_yticks(np.arange(100))
# # ... and label them with the respective list entries
# ax.set_xticklabels(range(10), fontsize=2)
# ax.set_yticklabels(range(100),fontsize=2)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(100):
#     for j in range(10):
#         text = ax.text(j, i, round(labels_heatmap_matrix[i][j]),
#                         ha="center", va="center", color="w", fontsize=2)

# ax.set_title("100 mnist imgs from 1000 img T-set Labels"+'\n'+
#              "special case C2 eval set",fontsize=6)
# fig.tight_layout()

# cbar = plt.colorbar(im)
# cbar.ax.tick_params(labelsize=4)

# # plt.savefig('100 from 1000 mnist images labels (C2 special case eval set) heatmap.png',dpi=1200)
# plt.show()


# #Softmax guesses next
# fig, ax = plt.subplots()
# im = ax.imshow(softmax_heatmap_matrix)

# # We want to show all ticks...
# ax.set_xticks(np.arange(10))
# ax.set_yticks(np.arange(100))
# # ... and label them with the respective list entries
# ax.set_xticklabels(range(10), fontsize=2)
# ax.set_yticklabels(range(100),fontsize=2)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#           rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(100):
#     for j in range(10):
#         text = ax.text(j, i, round(softmax_heatmap_matrix[i][j]),
#                         ha="center", va="center", color="w", fontsize=2)

# ax.set_title("288,288 C2 model eval of 100 mnist imgs from 1000 img T-set"+'\n'+
#               "softmax output (Uo.dot(r[n]))",fontsize=6)
# fig.tight_layout()

# cbar = plt.colorbar(im)
# cbar.ax.tick_params(labelsize=4)

# # plt.savefig('288 288 C2 model eval (of 100 from 1000)- softmax output heatmap.png',dpi=1200)
# plt.show()


# counter = 0

# for classif in Classif_success_by_img:
#     if classif == 0:
#         print('Classification img {} failed'.format(counter))
#         counter += 1
#     else:
#         print('Classification img {} succeeded'.format(counter))
#         counter += 1
        
# print('softmax_heatmap_matrix:')
# print(softmax_heatmap_matrix)

dig_ind1 = 0
dig_ind2 = 100

sum_dict = {}

for dig in range(0,10):
    set_of_one_dig = softmax_heatmap_matrix[dig_ind1:dig_ind2]
    example_counter = 1
    for example in set_of_one_dig:
        print('digit {} example {}:'.format(dig,example_counter))
        rounded_example = []
        for elem in example:
            rounded_elem = round(elem,3)
            rounded_example.append(rounded_elem)
        print(rounded_example)
        example_counter += 1
    dig_ind1 += 100
    dig_ind2 += 100
    sum_dict[dig] = 0
    
print('sum_dict pre-filled')
print(sum_dict)

for row in softmax_heatmap_matrix:
    sum_dict[np.argmax(row)] += 1

print('sum_dict final')
print(sum_dict)

