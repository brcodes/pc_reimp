from data import get_mnist_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random


"""
This script is for preprocessing and pickling image data for use with
PredictiveCodingClassifier.train() or .test()
"""

"""
Load MNIST images and prepare 100 images, 10 of each digit (0-9)
"""

# load data
# frac_samp 0.000166 = 10 images
# frac_samp 0.00166 = 100 images
# frac_samp 0.0166 = 1000 images
# frac_samp 0.166 = 10000 images
# frac_samp 1 = 60000 images
(X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=0.0166,return_test=True)

# number of initial training images
num_imgs = y_train.shape[0]
# print(num_imgs)

# make evenly distributed dataset of digits 0 - 9, 10 digits each
# the following logic, which proceeds down to the "rescaling, normalization" area of main.py
#   takes however many initial training images there are (1000 here), and extracts
#   ten of each digit, generating an evenly distributed 10 dig x 10 images (100 image) training set

X_dict = {}
y_dict = {}

for i in range(0,10):
    X_dict[i] = np.zeros(shape=(1,28,28))
    y_dict[i] = np.zeros(shape=(1,10))

# print(X_dict[0].shape)
# print(y_dict[0].shape)

for i in range(0,num_imgs):
    label = y_train[i,:]
    digit = np.nonzero(label)[0][0]
    X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
    y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))

# verify intermediate dictionary setup step

# print("dict key sizes")
# for i in range(0,10):
#     print(X_dict[i][1:,:,:].shape)

# print("dict num keys")
# print(len(X_dict))

X_dist = np.zeros(shape=(1,28,28))
y_dist = np.zeros(shape=(1,10))

# i is each unique digit, 0 - 9
# dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# b) only take first ten filled vectors from the value per key
for i in range(0,10):
    X_dist = np.vstack((X_dist, X_dict[i][1:11,:,:]))
    y_dist = np.vstack((y_dist, y_dict[i][1:11,:]))

# remove first empty vector of X_dist, y_dist to generate final image and label set
X_dist = X_dist[1:,:,:]
y_dist = y_dist[1:,:]

# verify array shape and presence & type of raw data of 10 dig x 10 imgs dataset

# print(X_dist.shape)
# print(y_dist.shape)
# for i in range(0,100):
#     print(y_dist[i])
# for i in range(0,12):
#     print(X_dist[i])

# visually verify full 10 dig by 10 imgs practice set by printing each img

# for i in range(88,X_dist.shape[0]):
#     plt.imshow(X_dist[i,:,:])
#     plt.show()

"""
Load MNIST images and prepare 1,000 images, 100 of each digit (0-9)
"""

# # load data
# # frac_samp 0.166 = 10000 images
# (X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=0.166,return_test=True)

# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)

# # make evenly distributed dataset of digits 0 - 9, 100 digits each

# X_dict = {}
# y_dict = {}

# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,28,28))
#     y_dict[i] = np.zeros(shape=(1,10))

# # print(X_dict[0].shape)
# # print(y_dict[0].shape)

# for i in range(0,num_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


# X_dist = np.zeros(shape=(1,28,28))
# y_dist = np.zeros(shape=(1,10))

# # i is each unique digit, 0 - 9
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first ten filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:101,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:101,:]))

# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]

# # verify array shape and presence & type of raw data of 100 dig x 10 imgs dataset

# # print(X_dist.shape)
# # print(y_dist.shape)
# # for i in range(0,101):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])

# # visually verify 100 dig by 10 imgs practice set by printing

# # for i in range(899,X_dist.shape[0]):
# #     plt.imshow(X_dist[i,:,:])
# #     plt.show()

"""
Load MNIST images and prepare 10,000 images, 1,000 of each digit (0-9)
"""

# # load data
# # frac_samp None or 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=1,return_test=True)

# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)

# # make evenly distributed dataset of digits 0 - 9, 1000 digits each

# X_dict = {}
# y_dict = {}

# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,28,28))
#     y_dict[i] = np.zeros(shape=(1,10))

# # print(X_dict[0].shape)
# # print(y_dict[0].shape)

# for i in range(0,num_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


# X_dist = np.zeros(shape=(1,28,28))
# y_dist = np.zeros(shape=(1,10))

# # i is each unique digit, 0 - 9
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first ten filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:1001,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:1001,:]))

# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]

# # verify array shape and presence & type of raw data of 1000 dig x 10 imgs dataset

# print(X_dist.shape)
# print(y_dist.shape)
# for i in range(0,1001):
#     print(y_dist[i])
# for i in range(0,12):
#     print(X_dist[i])

# # visually verify 1000 dig by 10 imgs practice set by printing

# # for i in range(8999,X_dist.shape[0]):
# #     plt.imshow(X_dist[i,:,:])
# #     plt.show()

"""
"Out of bag" (non-training) image, and "In bag" normal , "In bag" scrambled images
"""

# out of bag image
non_training_img = np.copy(X_test[0,:,:])

# # verify image
# plt.imshow(non_training_img, cmap='Greys'),plt.title('out of bag mnist image')
# plt.show()


# in bag image
training_img = np.copy(X_dist[0,:,:])
# in bag scrambled
scrambled = np.copy(X_dist[0,:,:])
scrambled = scrambled.ravel()
np.random.shuffle(scrambled)
scrambled = scrambled.reshape(28,28)

# # verify in bag normal and scrambled images
# plt.subplot(121), plt.imshow(training_img, cmap='Greys'),plt.title('in bag mnist normal')
# plt.subplot(122), plt.imshow(scrambled, cmap='Greys'),plt.title('in bag mnist scrambled')





"""
Lena loading and pre-processing
"""


lena_pw_path = 'non_mnist_images/lena_128x128_grey_prewhitened.png'
lena_zoom_path = 'non_mnist_images/lena_128x128_grey_zoomed.png'

# second arg of imread is 0 to denote reading in greyscale mode
lena_pw_read = cv2.imread(lena_pw_path,0)
lena_zoom_read = cv2.imread(lena_zoom_path,0)

lena_pw = cv2.resize(lena_pw_read,(28,28))
lena_zoom = cv2.resize(lena_zoom_read,(28,28))



# print(lena_pw)
# print(lena_zoom)


# # verify images
# plt.subplot(121), plt.imshow(lena_zoom, cmap='Greys'),plt.title('zoom')
# plt.subplot(122), plt.imshow(lena_pw, cmap='Greys'),plt.title('prewhitened')
# plt.show()



"""
Scaling and Pickling for Linear Model
"""

# NOTE: comment out code between these NOTES when pickling the Tanh dataset


# # standardize and flatten main database
# X_stdized = standardization_filter(X_dist)
# X_flat = flatten_images(X_stdized)
# # #verify stdization
# # plt.imshow(X_stdized[0,:,:], cmap='Greys'),plt.title('in bag standardized database')
# # plt.show()

# # stdize and flatten one (the first, [0,:,:]) image from main database
# training_img_std = standardization_filter(training_img[None,:,:])
# training_img_flat = flatten_images(training_img_std)
# # #verify stdization
# # training_img_sq = np.squeeze(training_img_std)
# # plt.imshow(training_img_sq, cmap='Greys'),plt.title('in bag standardized')
# # plt.show()

# # stdize/flatten out of bag image
# non_training_img_std = standardization_filter(non_training_img[None,:,:])
# non_training_img_flat = flatten_images(non_training_img_std)
# # #verify stdization
# # non_training_img_sq = np.squeeze(non_training_img_std)
# # plt.imshow(non_training_img_sq, cmap='Greys'),plt.title('out of bag standardized')
# # plt.show()

# # stdize/flatten in bag scrambled image
# scrambled_std = standardization_filter(scrambled[None,:,:])
# scrambled_flat = flatten_images(scrambled_std)
# # # verify stdization
# # scrambled_sq = np.squeeze(scrambled_std)
# # plt.imshow(scrambled_sq, cmap='Greys'),plt.title('scrambled standardized')
# # plt.show()

# # stdize/flatten lena prewhitened
# lena_pw_std = standardization_filter(lena_pw[None,:,:])
# lena_pw_flat = flatten_images(lena_pw_std)
# # # verify stdization
# # lena_pw_sq = np.squeeze(lena_pw_std)
# # plt.imshow(lena_pw_sq, cmap='Greys'),plt.title('lena prewhitened')
# # plt.show()

# # flatten lena zoomed for pickle output
# lena_zoom_std = standardization_filter(lena_zoom[None,:,:])
# lena_zoom_flat = flatten_images(lena_zoom_std)
# # # verify stdization
# # lena_zoom_sq = np.squeeze(lena_zoom_std)
# # plt.imshow(lena_zoom_sq, cmap='Greys'),plt.title('lena zoom')
# # plt.show()



# # pickle the flattened input images and the output vectors as a tuple
# linear_data_out = open('linear_10x10.pydb','wb')
# pickle.dump((X_flat, y_dist, training_img_flat, non_training_img_flat, scrambled_flat, lena_pw_flat, lena_zoom_flat), linear_data_out)
# linear_data_out.close()


# NOTE: comment out code between these NOTES when pickling the Tanh dataset


"""
Scaling and Pickling for Tanh Model
"""

# NOTE: comment out code between these NOTES when pickling the Linear dataset


# scale main database to [-1,1] and flatten
X_tanh = rescaling_filter(X_dist, scaling_range=[-1,1])
X_flat_tanh = flatten_images(X_tanh)
# # verify tanh scaling
# plt.imshow(X_tanh[0,:,:], cmap='Greys'),plt.title('in bag tanh database')
# plt.show()

# scale in bag image to [-1,1] and flatten
training_img_tanh = rescaling_filter(training_img, scaling_range=[-1,1])
training_img_tanh_flat = flatten_images(training_img_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(training_img_tanh, cmap='Greys'),plt.title('in bag tanh')
# plt.show()

# scale out of bag image to [-1,1] and flatten
non_training_img_tanh = rescaling_filter(non_training_img, scaling_range=[-1,1])
non_training_img_tanh_flat = flatten_images(non_training_img_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(non_training_img_tanh, cmap='Greys'),plt.title('out of bag tanh')
# plt.show()

# scale in bag scrambled to [-1,1] and flatten
scrambled_tanh = rescaling_filter(scrambled, scaling_range=[-1,1])
scrambled_tanh_flat = flatten_images(scrambled_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(scrambled_tanh, cmap='Greys'),plt.title('in bag scrambled tanh')
# plt.show()


# scale lena pw to [-1,1] and flatten
lena_pw_tanh = rescaling_filter(lena_pw, scaling_range=[-1,1])
lena_pw_tanh_flat = flatten_images(lena_pw_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(lena_pw_tanh, cmap='Greys'),plt.title('lena pw tanh')
# plt.show()


# scale lena zoom to [-1,1] and flatten
lena_zoom_tanh = rescaling_filter(lena_zoom, scaling_range=[-1,1])
lena_zoom_tanh_flat = flatten_images(lena_zoom_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(lena_zoom_tanh, cmap='Greys'),plt.title('lena zoom tanh')
# plt.show()



# plt.hist(training_img_tanh.ravel(),256,[-1,1])
# plt.suptitle("training image tanh histogram")
# axes = plt.gca()
# axes.set_ylim(0,784)
# plt.xlabel('pixel intensity value (-1,1)')
# plt.ylabel('number of pixels (0-784)')
# plt.show()

# plt.hist(non_training_img_tanh.ravel(),256,[-1,1])
# plt.suptitle("non training image tanh histogram")
# axes = plt.gca()
# axes.set_ylim(0,784)
# plt.xlabel('pixel intensity value (-1,1)')
# plt.ylabel('number of pixels (0-784)')
# plt.show()

# plt.hist(scrambled_tanh.ravel(),256,[-1,1])
# plt.suptitle("scrambled training image tanh histogram")
# axes = plt.gca()
# axes.set_ylim(0,784)
# plt.xlabel('pixel intensity value (-1,1)')
# plt.ylabel('number of pixels (0-784)')
# plt.show()

# plt.hist(lena_zoom_tanh.ravel(),256,[-1,1])
# plt.suptitle("lena zoomed image tanh histogram")
# axes = plt.gca()
# axes.set_ylim(0,784)
# plt.xlabel('pixel intensity value (-1,1)')
# plt.ylabel('number of pixels (0-784)')
# plt.show()

# plt.hist(lena_pw_tanh.ravel(),256,[-1,1])
# plt.suptitle("lena prewhitened image tanh histogram")
# axes = plt.gca()
# axes.set_ylim(0,784)
# plt.xlabel('pixel intensity value (-1,1)')
# plt.ylabel('number of pixels (0-784)')
# plt.show()




# pickle the flattened input images and the output vectors as a tuple
tanh_data_out = open('tanh_100x10.pydb', 'wb')
pickle.dump((X_flat_tanh, y_dist, training_img_tanh_flat, non_training_img_tanh_flat, scrambled_tanh_flat, lena_pw_tanh_flat, lena_zoom_tanh_flat), tanh_data_out)
tanh_data_out.close()


# NOTE: comment out code between these NOTES when pickling the Linear dataset
