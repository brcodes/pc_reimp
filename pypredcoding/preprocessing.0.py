 from data import get_mnist_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors, diff_of_gaussians_filter
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random


"""
This script is for preprocessing and pickling image data for use with
PredictiveCodingClassifier.train(), evaluate(), or predict()

or with

TiledPredictiveCodingClassifier.train(), evaluate(), predict()
"""


"""
Load MNIST images and prepare 100 images, 10 of each digit (0-9)

100 NORMAL 28x28 images

"""
#
# # load data
# # frac_samp 0.000166 = 10 images
# # frac_samp 0.00166 = 100 images
# # frac_samp 0.0166 = 1000 images
# # frac_samp 0.166 = 10000 images
# # frac_samp 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=0.0166,return_test=True)
#
# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)
#
# # make evenly distributed dataset of digits 0 - 9, 10 digits each
# # the following logic, which proceeds down to the "rescaling, normalization" area of main.py
# #   takes however many initial training images there are (1000 here), and extracts
# #   ten of each digit, generating an evenly distributed 10 dig x 10 images (100 image) training set
#
# X_dict = {}
# y_dict = {}
#
# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,28,28))
#     y_dict[i] = np.zeros(shape=(1,10))
#
# # print(X_dict[0].shape)
# # print(y_dict[0].shape)
#
# for i in range(0,num_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))
#
# # verify intermediate dictionary setup step
#
# # print("dict key sizes")
# # for i in range(0,10):
# #     print(X_dict[i][1:,:,:].shape)
#
# # print("dict num keys")
# # print(len(X_dict))
#
# X_dist = np.zeros(shape=(1,28,28))
# y_dist = np.zeros(shape=(1,10))
#
# # i is each unique digit, 0 - 9
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first ten filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:11,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:11,:]))
#
# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]
#
# # verify array shape and presence & type of raw data of 10 dig x 10 imgs dataset
#
# # print(X_dist.shape)
# # print(y_dist.shape)
# # for i in range(0,100):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])
#
# # visually verify full 10 dig by 10 imgs practice set by printing each img
#
# # for i in range(88,X_dist.shape[0]):
# #     plt.imshow(X_dist[i,:,:])
# #     plt.show()

"""
Load MNIST images and prepare 100 images, 10 of each digit (0-9)

100 DOWNSAMPLED 24x24 images

"""

# # load data
# # frac_samp 0.000166 = 10 images
# # frac_samp 0.00166 = 100 images
# # frac_samp 0.0166 = 1000 images
# # frac_samp 0.166 = 10000 images
# # frac_samp 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=0.0166,return_test=True)

# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)

# # make evenly distributed dataset of digits 0 - 9, 10 digits each
# # the following logic, which proceeds down to the "rescaling, normalization" area of main.py
# #   takes however many initial training images there are (1000 here), and extracts
# #   ten of each digit, generating an evenly distributed 10 dig x 10 images (100 image) training set

# X_dict = {}
# y_dict = {}

# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,24,24))
#     y_dict[i] = np.zeros(shape=(1,10))

# # print(X_dict[0].shape)
# # print(y_dict[0].shape)

# for i in range(0,num_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     # the three following lines are all that change in the main downsampling image loop
#     image = X_train[i,:,:]
#     resized_image = cv2.resize(image,(24,24))
#     X_dict[digit] = np.vstack((X_dict[digit], resized_image[None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))

# # verify intermediate dictionary setup step

# # print("dict key sizes")
# # for i in range(0,10):
# #     print(X_dict[i][1:,:,:].shape)

# # print("dict num keys")
# # print(len(X_dict))

# X_dist = np.zeros(shape=(1,24,24))
# y_dist = np.zeros(shape=(1,10))

# # i is each unique digit, 0 - 9
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first ten filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:11,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:11,:]))

# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]

# # verify array shape and presence & type of raw data of 10 dig x 10 imgs dataset

# # print(X_dist.shape)
# # print(y_dist.shape)
# # for i in range(0,100):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])

# # visually verify full 10 dig by 10 imgs practice set by printing each img

# print('X_dist[0].shape is {}'.format(X_dist[0].shape))

# # for i in range(88,X_dist.shape[0]):
# #     plt.imshow(X_dist[i,:,:])
# #     plt.show()

"""
Load MNIST images and prepare 1,000 images, 100 of each digit (0-9)

1,000 NORMAL 28x28 images

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
Load MNIST images and prepare 1,000 images, 100 of each digit (0-9)

1,000 DOWNSAMPLED 24x24 images

"""

# # load data
# # frac_samp 0.166 = 10000 images
# (X_train, y_train),(X_test,y_test) = get_mnist_data(frac_samp=0.166,return_test=True)
#
# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)
#
# # make evenly distributed dataset of digits 0 - 9, 100 digits each
#
# X_dict = {}
# y_dict = {}
#
# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,24,24))
#     y_dict[i] = np.zeros(shape=(1,10))
#
# # print(X_dict[0].shape)
# # print(y_dict[0].shape)
#
# for i in range(0,num_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     image = X_train[i,:,:]
#     resized_image = cv2.resize(image,(24,24))
#     X_dict[digit] = np.vstack((X_dict[digit], resized_image[None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))
#
#
# X_dist = np.zeros(shape=(1,24,24))
# y_dist = np.zeros(shape=(1,10))
#
# # i is each unique digit, 0 - 9
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first ten filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:101,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:101,:]))
#
# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]
#
# # verify array shape and presence & type of raw data of 100 dig x 10 imgs dataset
#
# # print(X_dist.shape)
# # print(y_dist.shape)
# # for i in range(0,101):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])
#
# # visually verify 100 dig by 10 imgs practice set by printing
#
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
# non_training_img = np.copy(X_test[0,:,:])

# for 24x24 image
# non_training_img = cv2.resize(non_training_img, (24,24))

# # verify image
# plt.imshow(non_training_img, cmap='Greys'),plt.title('out of bag mnist image')
# plt.show()


# # in bag image
# training_img = np.copy(X_dist[0,:,:])
# # in bag scrambled
# scrambled = np.copy(X_dist[0,:,:])
# scrambled = scrambled.ravel()
# np.random.shuffle(scrambled)

# # # for 28x28 image
# scrambled = scrambled.reshape(28,28)

# for 24x24 image
# scrambled = scrambled.reshape(24,24)

# # verify in bag normal and scrambled images
# plt.subplot(121), plt.imshow(training_img, cmap='Greys'),plt.title('in bag mnist normal')
# plt.subplot(122), plt.imshow(scrambled, cmap='Greys'),plt.title('in bag mnist scrambled')





"""
Lena loading and pre-processing
"""


# lena_pw_path = 'non_mnist_images/lena_128x128_grey_prewhitened.png'
# lena_zoom_path = 'non_mnist_images/lena_128x128_grey_zoomed.png'

# # second arg of imread is 0 to denote reading in greyscale mode
# lena_pw_read = cv2.imread(lena_pw_path,0)
# lena_zoom_read = cv2.imread(lena_zoom_path,0)

# # lena_pw = cv2.resize(lena_pw_read,(28,28))
# # lena_zoom = cv2.resize(lena_zoom_read,(28,28))

# # for 24x24
# lena_pw = cv2.resize(lena_pw_read,(24,24))
# lena_zoom = cv2.resize(lena_zoom_read,(24,24))


# print(lena_pw)
# print(lena_zoom)


# # verify images
# plt.subplot(121), plt.imshow(lena_zoom, cmap='Greys'),plt.title('zoom')
# plt.subplot(122), plt.imshow(lena_pw, cmap='Greys'),plt.title('prewhitened')
# plt.show()

'''
rao images
'''

#From Rao 1999 Vision Research paper

rao_vr_pepsi_path = 'non_mnist_images/images_rao_visionres/pepsi.png'
rao_vr_bear_path = 'non_mnist_images/images_rao_visionres/bear.png'
rao_vr_spraycan_path = 'non_mnist_images/images_rao_visionres/spraycan.png'
rao_vr_doll_path = 'non_mnist_images/images_rao_visionres/doll.png'
rao_vr_teapot_path = 'non_mnist_images/images_rao_visionres/teapot.png'

rao_vr_pepsi_read = cv2.imread(rao_vr_pepsi_path,0)
rao_vr_bear_read = cv2.imread(rao_vr_bear_path,0)
rao_vr_spraycan_read = cv2.imread(rao_vr_spraycan_path,0)
rao_vr_doll_read = cv2.imread(rao_vr_doll_path,0)
rao_vr_teapot_read = cv2.imread(rao_vr_teapot_path,0)

rao_vr_pepsi = cv2.resize(rao_vr_pepsi_read,(28,28))
rao_vr_bear = cv2.resize(rao_vr_bear_read,(28,28))
rao_vr_spraycan = cv2.resize(rao_vr_spraycan_read,(28,28))
rao_vr_doll = cv2.resize(rao_vr_doll_read,(28,28))
rao_vr_teapot = cv2.resize(rao_vr_teapot_read,(28,28))

#From Rao and Ballard 1999 Nature paper

rb_nat_monkey_path = 'non_mnist_images/images_rao_nature/monkey.png'
rb_nat_swan_path = 'non_mnist_images/images_rao_nature/swan.png'
rb_nat_rose_path = 'non_mnist_images/images_rao_nature/rose.png'
rb_nat_zebra_path = 'non_mnist_images/images_rao_nature/zebra.png'
rb_nat_forest_path = 'non_mnist_images/images_rao_nature/forest.png'

rb_nat_monkey_read = cv2.imread(rb_nat_monkey_path,0)
rb_nat_swan_read = cv2.imread(rb_nat_swan_path,0) 
rb_nat_rose_read = cv2.imread(rb_nat_rose_path,0)
rb_nat_zebra_read = cv2.imread(rb_nat_zebra_path,0)
rb_nat_forest_read = cv2.imread(rb_nat_forest_path,0)

rb_nat_monkey = cv2.resize(rb_nat_monkey_read,(28,28))
rb_nat_swan = cv2.resize(rb_nat_swan_read,(28,28))
rb_nat_rose = cv2.resize(rb_nat_rose_read,(28,28))
rb_nat_zebra = cv2.resize(rb_nat_zebra_read,(28,28))
rb_nat_forest = cv2.resize(rb_nat_forest_read,(28,28))



"""
Scaling and Pickling for Linear Model
"""

# NOTE: comment out code between these NOTES when pickling the Tanh dataset

#
# # standardize and flatten main database
# X_stdized = standardization_filter(X_dist)
# X_flat = flatten_images(X_stdized)
# # #verify stdization
# plt.imshow(X_stdized[0,:,:], cmap='Greys'),plt.title('in bag standardized database')
# plt.show()
#
# # stdize and flatten one (the first, [0,:,:]) image from main database
# training_img_std = standardization_filter(training_img[None,:,:])
# training_img_flat = flatten_images(training_img_std)
# # #verify stdization
# training_img_sq = np.squeeze(training_img_std)
# plt.imshow(training_img_sq, cmap='Greys'),plt.title('in bag standardized')
# plt.show()
#
# # stdize/flatten out of bag image
# non_training_img_std = standardization_filter(non_training_img[None,:,:])
# non_training_img_flat = flatten_images(non_training_img_std)
# # #verify stdization
# non_training_img_sq = np.squeeze(non_training_img_std)
# plt.imshow(non_training_img_sq, cmap='Greys'),plt.title('out of bag standardized')
# plt.show()
#
# # stdize/flatten in bag scrambled image
# scrambled_std = standardization_filter(scrambled[None,:,:])
# scrambled_flat = flatten_images(scrambled_std)
# # # verify stdization
# scrambled_sq = np.squeeze(scrambled_std)
# plt.imshow(scrambled_sq, cmap='Greys'),plt.title('scrambled standardized')
# plt.show()
#
# # stdize/flatten lena prewhitened
# lena_pw_std = standardization_filter(lena_pw[None,:,:])
# lena_pw_flat = flatten_images(lena_pw_std)
# # # verify stdization
# lena_pw_sq = np.squeeze(lena_pw_std)
# plt.imshow(lena_pw_sq, cmap='Greys'),plt.title('lena prewhitened')
# plt.show()
#
# # flatten lena zoomed for pickle output
# lena_zoom_std = standardization_filter(lena_zoom[None,:,:])
# lena_zoom_flat = flatten_images(lena_zoom_std)
# # # verify stdization
# lena_zoom_sq = np.squeeze(lena_zoom_std)
# plt.imshow(lena_zoom_sq, cmap='Greys'),plt.title('lena zoom')
# plt.show()
#
#
# # test out
#
#
# # pickle the flattened input images and the output vectors as a tuple
# linear_data_out = open('linear_100x10_size_24x24.pydb','wb')
# pickle.dump((X_flat, y_dist, training_img_flat, non_training_img_flat, scrambled_flat, lena_pw_flat, lena_zoom_flat), linear_data_out)
# linear_data_out.close()


# NOTE: comment out code between these NOTES when pickling the Tanh dataset


"""
Scaling and Pickling for Tanh Model
"""

# # NOTE: comment out code between these NOTES when pickling the Linear dataset


# # scale main database to [-1,1] and flatten
# X_tanh = rescaling_filter(X_dist, scaling_range=[-1,1])
# X_flat_tanh = flatten_images(X_tanh)
# # # verify tanh scaling
# plt.imshow(X_tanh[0,:,:], cmap='Greys'),plt.title('in bag tanh database')
# plt.show()

# # scale in bag image to [-1,1] and flatten
# training_img_tanh = rescaling_filter(training_img, scaling_range=[-1,1])
# training_img_tanh_flat = flatten_images(training_img_tanh[None,:,:])
# # # verify tanh scaling
# plt.imshow(training_img_tanh, cmap='Greys'),plt.title('in bag tanh')
# plt.show()

# # scale out of bag image to [-1,1] and flatten
# non_training_img_tanh = rescaling_filter(non_training_img, scaling_range=[-1,1])
# non_training_img_tanh_flat = flatten_images(non_training_img_tanh[None,:,:])
# # # verify tanh scaling
# plt.imshow(non_training_img_tanh, cmap='Greys'),plt.title('out of bag tanh')
# plt.show()

# # scale in bag scrambled to [-1,1] and flatten
# scrambled_tanh = rescaling_filter(scrambled, scaling_range=[-1,1])
# scrambled_tanh_flat = flatten_images(scrambled_tanh[None,:,:])
# # # verify tanh scaling
# plt.imshow(scrambled_tanh, cmap='Greys'),plt.title('in bag scrambled tanh')
# plt.show()


# # scale lena pw to [-1,1] and flatten
# lena_pw_tanh = rescaling_filter(lena_pw, scaling_range=[-1,1])
# lena_pw_tanh_flat = flatten_images(lena_pw_tanh[None,:,:])
# # # verify tanh scaling
# plt.imshow(lena_pw_tanh, cmap='Greys'),plt.title('lena pw tanh')
# plt.show()


# # scale lena zoom to [-1,1] and flatten
# lena_zoom_tanh = rescaling_filter(lena_zoom, scaling_range=[-1,1])
# lena_zoom_tanh_flat = flatten_images(lena_zoom_tanh[None,:,:])
# # # verify tanh scaling
# plt.imshow(lena_zoom_tanh, cmap='Greys'),plt.title('lena zoom tanh')
# plt.show()


'''
rao images
'''

# 1999 VisionRes Paper

# scale  to [-1,1] and flatten
rao_vr_pepsi_tanh = rescaling_filter(rao_vr_pepsi, scaling_range=[-1,1])
rao_vr_pepsi_tanh_flat = flatten_images(rao_vr_pepsi_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(rao_vr_pepsi_tanh, cmap='gray'),plt.title('rao visionres pepsi tanh')
# plt.show()

# scale  to [-1,1] and flatten
rao_vr_bear_tanh = rescaling_filter(rao_vr_bear, scaling_range=[-1,1])
rao_vr_bear_tanh_flat = flatten_images(rao_vr_bear_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(rao_vr_bear_tanh, cmap='gray'),plt.title('rao visionres bear tanh')
# plt.show()

# scale  to [-1,1] and flatten
rao_vr_spraycan_tanh = rescaling_filter(rao_vr_spraycan, scaling_range=[-1,1])
rao_vr_spraycan_tanh_flat = flatten_images(rao_vr_spraycan_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(rao_vr_spraycan_tanh, cmap='gray'),plt.title('rao visionres spraycan tanh')
# plt.show()

# scale  to [-1,1] and flatten
rao_vr_doll_tanh = rescaling_filter(rao_vr_doll, scaling_range=[-1,1])
rao_vr_doll_tanh_flat = flatten_images(rao_vr_doll_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(rao_vr_doll_tanh, cmap='gray'),plt.title('rao visionres doll tanh')
# plt.show()

# scale  to [-1,1] and flatten
rao_vr_teapot_tanh = rescaling_filter(rao_vr_teapot, scaling_range=[-1,1])
rao_vr_teapot_tanh_flat = flatten_images(rao_vr_teapot_tanh[None,:,:])
# # verify tanh scaling
# plt.imshow(rao_vr_teapot_tanh, cmap='gray'),plt.title('rao visionres teapot tanh')
# plt.show()


# 1999 Nature Paper

# # standardize and flatten main database
# X_stdized = standardization_filter(X_dist)
# X_flat = flatten_images(X_stdized)
# # #verify stdization
# plt.imshow(X_stdized[0,:,:], cmap='Greys'),plt.title('in bag standardized database')
# plt.show()


# scale  to [-1,1] and flatten
rb_nat_monkey_tanh_flat = flatten_images(rb_nat_monkey[None,:,:])
# # verify tanh scaling
plt.imshow(np.squeeze(rb_nat_monkey), cmap='gray'),plt.title('r&b nature monkey lin')
plt.show()

# scale  to [-1,1] and flatten
rb_nat_swan_tanh_flat = flatten_images(rb_nat_swan[None,:,:])
# # verify tanh scaling
plt.imshow(np.squeeze(rb_nat_swan), cmap='gray'),plt.title('r&b nature swan lin')
plt.show()

# scale  to [-1,1] and flatten
rb_nat_rose_tanh_flat = flatten_images(rb_nat_rose[None,:,:])
# # verify tanh scaling
plt.imshow(np.squeeze(rb_nat_rose), cmap='gray'),plt.title('r&b nature rose lin')
plt.show()

# scale  to [-1,1] and flatten
rb_nat_zebra_tanh_flat = flatten_images(rb_nat_zebra[None,:,:])
# # verify tanh scaling
plt.imshow(np.squeeze(rb_nat_zebra), cmap='gray'),plt.title('r&b nature zebra lin')
plt.show()

# scale  to [-1,1] and flatten
rb_nat_forest_tanh_flat = flatten_images(rb_nat_forest[None,:,:])
# # verify tanh scaling
plt.imshow(np.squeeze(rb_nat_forest), cmap='gray'),plt.title('r&b nature forest lin')
plt.show()




# # scale  to [-1,1] and flatten
# rb_nat_monkey_std = standardization_filter(rb_nat_monkey[None,:,:])
# rb_nat_monkey_tanh_flat = flatten_images(rb_nat_monkey_std)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_monkey_std), cmap='gray'),plt.title('r&b nature monkey lin')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_swan_std = standardization_filter(rb_nat_swan[None,:,:])
# rb_nat_swan_tanh_flat = flatten_images(rb_nat_swan_std)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_swan_std), cmap='gray'),plt.title('r&b nature swan lin')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_rose_std = standardization_filter(rb_nat_rose[None,:,:])
# rb_nat_rose_tanh_flat = flatten_images(rb_nat_rose_std)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_rose_std), cmap='gray'),plt.title('r&b nature rose lin')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_zebra_std = standardization_filter(rb_nat_zebra[None,:,:])
# rb_nat_zebra_tanh_flat = flatten_images(rb_nat_zebra_std)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_zebra_std), cmap='gray'),plt.title('r&b nature zebra lin')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_forest_std = standardization_filter(rb_nat_forest[None,:,:])
# rb_nat_forest_tanh_flat = flatten_images(rb_nat_forest_std)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_forest_std), cmap='gray'),plt.title('r&b nature forest lin')
# plt.show()



# # scale  to [-1,1] and flatten
# rb_nat_monkey_dog = diff_of_gaussians_filter(rb_nat_monkey[None,:,:])
# rb_nat_monkey_tanh = rescaling_filter(rb_nat_monkey_dog, scaling_range=[-1,1])
# rb_nat_monkey_tanh_flat = flatten_images(rb_nat_monkey_tanh)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_monkey_tanh), cmap='gray'),plt.title('r&b nature monkey tanh dog')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_swan_dog = diff_of_gaussians_filter(rb_nat_swan[None,:,:])
# rb_nat_swan_tanh = rescaling_filter(rb_nat_swan_dog, scaling_range=[-1,1])
# rb_nat_swan_tanh_flat = flatten_images(rb_nat_swan_tanh)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_swan_tanh), cmap='gray'),plt.title('r&b nature swan tanh dog')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_rose_dog = diff_of_gaussians_filter(rb_nat_rose[None,:,:])
# rb_nat_rose_tanh = rescaling_filter(rb_nat_rose_dog, scaling_range=[-1,1])
# rb_nat_rose_tanh_flat = flatten_images(rb_nat_rose_tanh)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_rose_tanh), cmap='gray'),plt.title('r&b nature rose tanh dog')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_zebra_dog = diff_of_gaussians_filter(rb_nat_zebra[None,:,:])
# rb_nat_zebra_tanh = rescaling_filter(rb_nat_zebra_dog, scaling_range=[-1,1])
# rb_nat_zebra_tanh_flat = flatten_images(rb_nat_zebra_tanh)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_zebra_tanh), cmap='gray'),plt.title('r&b nature zebra tanh dog')
# plt.show()

# # scale  to [-1,1] and flatten
# rb_nat_forest_dog = diff_of_gaussians_filter(rb_nat_forest[None,:,:])
# rb_nat_forest_tanh = rescaling_filter(rb_nat_forest_dog, scaling_range=[-1,1])
# rb_nat_forest_tanh_flat = flatten_images(rb_nat_forest_tanh)
# # # verify tanh scaling
# plt.imshow(np.squeeze(rb_nat_forest_tanh), cmap='gray'),plt.title('r&b nature forest tanh dog')
# plt.show()



rao_vr_img_list = [rao_vr_pepsi_tanh_flat,rao_vr_bear_tanh_flat,rao_vr_spraycan_tanh_flat,rao_vr_doll_tanh_flat,rao_vr_teapot_tanh_flat]
rb_nat_img_list = [rb_nat_monkey_tanh_flat,rb_nat_swan_tanh_flat,rb_nat_rose_tanh_flat,rb_nat_zebra_tanh_flat,rb_nat_forest_tanh_flat]


combined_vr_imgs_vec = np.zeros(shape=(1,784))
combined_nat_imgs_vec = np.zeros(shape=(1,784))
combined_labels_vec = np.zeros(shape=(1,5))



for i in range(0,5):
    #image parsing and stacking
    vr_img = rao_vr_img_list[i]
    nat_img = rb_nat_img_list[i]
    reshaped_vr = vr_img.reshape(1,784)
    reshaped_nat = nat_img.reshape(1,784)
    combined_vr_imgs_vec = np.vstack((combined_vr_imgs_vec, reshaped_vr))
    combined_nat_imgs_vec = np.vstack((combined_nat_imgs_vec, reshaped_nat))
    
    #label creation and stacking
    label = np.zeros(shape=(1,5))
    label[:,i] = 1
    combined_labels_vec = np.vstack((combined_labels_vec,label))
    
print('final shape of combined vr imgs vec is {}'.format(combined_vr_imgs_vec.shape))
print('final shape of combined nat imgs vec is {}'.format(combined_nat_imgs_vec.shape))
print('final shape of combined labels vec is {}'.format(combined_labels_vec.shape))
print('final combined labels vec is')
print(combined_labels_vec)
    
    
vr_imgs_vec = combined_vr_imgs_vec[1:6]
nat_imgs_vec = combined_nat_imgs_vec[1:6]
labels_vec = combined_labels_vec[1:6]
    
print('final shape of vr imgs vec is {}'.format(vr_imgs_vec.shape))
print('final shape of nat imgs vec is {}'.format(nat_imgs_vec.shape))
print('final labels vec is')
print(labels_vec)
    
'''
Plot histograms of "increasingly out-of-bag images 1-5"
'''

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


'''
Pickle out whatever tanh dataset has been created above
'''

# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('tanh_10x10_size_24x24.pydb', 'wb')
# pickle.dump((X_flat_tanh, y_dist, training_img_tanh_flat, non_training_img_tanh_flat, scrambled_tanh_flat, lena_pw_tanh_flat, lena_zoom_tanh_flat), tanh_data_out)
# tanh_data_out.close()

# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('rao_ballard_nature_dog.pydb', 'wb')
# pickle.dump((nat_imgs_vec,labels_vec), tanh_data_out)
# tanh_data_out.close()

# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('rao_ballard_nature_lin.pydb', 'wb')
# pickle.dump((nat_imgs_vec,labels_vec), tanh_data_out)
# tanh_data_out.close()

# pickle the flattened input images and the output vectors as a tuple
tanh_data_out = open('rao_ballard_nature_no_pre.pydb', 'wb')
pickle.dump((nat_imgs_vec,labels_vec), tanh_data_out)
tanh_data_out.close()

# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('rao_visionres_size_24x24.pydb', 'wb')
# pickle.dump((vr_imgs_vec, labels_vec), tanh_data_out)
# tanh_data_out.close()

'''
Take ten random examples of each digit from tanh mnist 100x10
'''

# index1 = 0
# index2 = 100

# X_distilled = np.zeros(shape=(1,784))
# y_distilled = np.zeros(shape=(1,10))

# for digit in range(0,10):
    
#     hundred_imgs_one_dig = X_flat_tanh[index1:index2]
#     print('size hundred_imgs_one_dig is {}'.format(hundred_imgs_one_dig.shape))
#     hundred_labels_one_dig = y_dist[index1:index2]
#     print('size hundred_labels_one_dig is {}'.format(hundred_labels_one_dig.shape))
#     ten_rand_indices = random.sample(range(100), 10)
#     print('ten rand indices for digit {} are {}'.format(digit,ten_rand_indices))
#     ten_imgs = np.zeros(shape=(1,784))
#     ten_labels = np.zeros(shape=(1,10))
#     for index in ten_rand_indices:
#         #Size 784
#         rand_img = hundred_imgs_one_dig[index]
#         print('size of rand_img is {}'.format(rand_img.shape))
#         #Size 10
#         rand_label = hundred_labels_one_dig[index]
#         print('size of rand_label is {}'.format(rand_label.shape))
#         ten_imgs = np.vstack((ten_imgs, rand_img[None,:]))
#         ten_labels = np.vstack((ten_labels, rand_label[None,:]))
#     print('size of ten_imgs after 10 rand insertions is {}'.format(ten_imgs.shape))
#     print('size of ten_labels after 10 rand insertions is {}'.format(ten_labels.shape))

#     #taking off first empty row
    
#     ten_imgs_squeezed = ten_imgs[1:11,:]
#     ten_labels_squeezed = ten_labels[1:11,:]
#     print('size of ten_imgs_squeezed is {}'.format(ten_imgs_squeezed.shape))
#     print('size of ten_labels_squeezed is {}'.format(ten_labels_squeezed.shape))
    
#     X_distilled = np.vstack((X_distilled, ten_imgs_squeezed))
#     y_distilled = np.vstack((y_distilled, ten_labels_squeezed))
    
#     print('size of X_distilled in loop is {}'.format(X_distilled.shape))
#     print('size of y_distilled in loop is {}'.format(y_distilled.shape))

#     index1 += 100
#     index2 += 100

# X_distilled_squeezed = X_distilled[1:101,:]
# y_distilled_squeezed = y_distilled[1:101,:]

# print('size of X_dist_squeezed after loop is {}'.format(X_distilled_squeezed.shape))
# print('size of y_dist_squeezed after loop is {}'.format(y_distilled_squeezed.shape))

# for i in range(0,100):
#     reshaped = X_distilled_squeezed[i].reshape(28,28)
#     plt.imshow(reshaped)
#     plt.show()
#     print(np.argmax(y_distilled_squeezed[i,:]))


# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('ten_of_each_dig_from_mnist_1000.pydb', 'wb')
# pickle.dump((X_distilled_squeezed, y_distilled_squeezed), tanh_data_out)
# tanh_data_out.close()

