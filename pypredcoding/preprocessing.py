from data import get_keras_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random
from skimage import color


"""
This script is for preprocessing and pickling image data for use with
PredictiveCodingClassifier.train(), evaluate(), or predict()

or with

TiledPredictiveCodingClassifier.train(), evaluate(), predict()
"""


"""
Load MNIST images and prepare 100 images, 10 of each digit (0-9)

100 NORMAL 28x28 images (for non-tiled model)

"""
#
# # load data
# # frac_samp 0.000166 = 10 images
# # frac_samp 0.00166 = 100 images
# # frac_samp 0.0166 = 1000 images
# # frac_samp 0.166 = 10000 images
# # frac_samp 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=0.0166,return_test=True)
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

100 DOWNSAMPLED 24x24 images (for tiled model)

"""

# # load data
# # frac_samp 0.000166 = 10 images
# # frac_samp 0.00166 = 100 images
# # frac_samp 0.0166 = 1000 images
# # frac_samp 0.166 = 10000 images
# # frac_samp 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=0.0166,return_test=True)

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

1,000 NORMAL 28x28 images (for non-tiled model)

"""

# # load data
# # frac_samp 0.166 = 10000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=0.166,return_test=True)

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
Load FASHION MNIST images and prepare 1,000 images, 100 of each class

1,000 NORMAL 28x28 images (for non-tiled model)

"""

# load data
# frac_samp 0.16666 = 10000 images
(X_train, y_train),(X_test,y_test) = get_keras_data(dataset='fashion_mnist',frac_samp=0.16666,return_test=True)

# number of initial training images
num_train_imgs = X_train.shape[0]
print(num_train_imgs)

# make evenly distributed dataset of classes 1-10, 100 imgs each
# 1 t-shirt/top
# 2 trouser
# 3 pullover
# 4 dress
# 5 coat
# 6 sandal
# 7 shirt
# 8 sneaker
# 9 bag
# 10 ankle boot


X_dict = {}
y_dict = {}

for i in range(0,10):
    X_dict[i] = np.zeros(shape=(1,28,28))
    y_dict[i] = np.zeros(shape=(1,10))

# print(X_dict[0].shape)
# print(y_dict[0].shape)

for i in range(0,num_train_imgs):
    label = y_train[i,:]
    digit = np.nonzero(label)[0][0]
    X_dict[digit] = np.vstack((X_dict[digit], X_train[i,:,:][None,:,:]))
    y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


X_dist = np.zeros(shape=(1,28,28))
y_dist = np.zeros(shape=(1,10))

# i is each unique class, 
# dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# b) only take first 100 filled vectors from the value per key
for i in range(0,10):
    X_dist = np.vstack((X_dist, X_dict[i][1:101,:,:]))
    y_dist = np.vstack((y_dist, y_dict[i][1:101,:]))

# remove first empty vector of X_dist, y_dist to generate final image and label set
X_dist = X_dist[1:,:,:]
y_dist = y_dist[1:,:]

# verify array shape and presence & type of raw data of 100 imgs x 10 imgs dataset

# print(X_dist.shape)
# print(y_dist.shape)
# for i in range(0,101):
#     print(y_dist[i])
# for i in range(0,12):
#     print(X_dist[i])

# visually verify 100 img by 10 classes practice set by printing

# for i in range(899,X_dist.shape[0]):
#     plt.imshow(X_dist[i,:,:],cmap='Greys')
#     plt.show()


"""
Load CIFAR-10 images and prepare 1,000 images, 100 of each class

1,000 NORMAL 28x28 images (for non-tiled model)

"""

# # load data
# # frac_samp 0.2 = 10000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='cifar10',frac_samp=0.2,return_test=True)


# print('X_train.shape')
# print(X_train.shape)
# print('X_test.shape')
# print(X_test.shape)


# # number of initial training images
# num_train_imgs = X_train.shape[0]
# num_test_imgs = X_test.shape[0]
# print(num_train_imgs)
# print(num_test_imgs)

# X_train_new = np.zeros(shape=(num_train_imgs,28,28))
# X_test_new = np.zeros(shape=(num_test_imgs,28,28))

# #Resize 32x32 training imgs in to 28x28
# for train_img in range(0,num_train_imgs):
#     print('\n')
#     print('train_img {}'.format(train_img+1))
#     print('\n')
#     #32x32
#     train_image = X_train[train_img,:,:]
#     # print('train_image.shape')
#     # print(train_image.shape)
#     grey_train_image = color.rgb2gray(train_image)
#     # grey_train_image = cv2.cvtColor(recolored_train_image,cv2.COLOR_BGR2GRAY)
#     # print('grey_train_image.shape')
#     # print(grey_train_image.shape)
#     #28x28
#     train_resized = cv2.resize(grey_train_image,(28,28))
#     # print('train_resized.shape')
#     # print(train_resized.shape)
#     #Put back in X_train
#     X_train_new[train_img] = train_resized

#     # print('X_train_new.shape after grey,28 processing and array loading')
#     # print(X_train_new.shape)
    
# #Resize 32x32 test imgs in to 28x28
# for test_img in range(0,num_test_imgs):
#     print('\n')
#     print('test_img {}'.format(test_img+1))
#     print('\n')
#     #32x32
#     test_image = X_test[test_img,:,:]
#     grey_test_image = color.rgb2gray(test_image)
#     # grey_test_image = cv2.cvtColor(recolored_test_image,cv2.COLOR_BGR2GRAY)
#     # print('grey_test_image.shape')
#     # print(grey_test_image.shape)
#     #28x28
#     test_resized = cv2.resize(grey_test_image,(28,28))
#     # print('test_resized.shape')
#     # print(test_resized.shape)
#     #Put back in X_train,X_test
#     X_test_new[test_img] = test_resized
    
#     # print('X_test_new.shape after grey,28 processing and array loading')
#     # print(X_test_new.shape)
    

# # make evenly distributed dataset of classes 1-10, 100 imgs each
# # 1 airplane
# # 2 car
# # 3 bird
# # 4 cat
# # 5 deer
# # 6 dog
# # 7 frog
# # 8 horse
# # 9 ship
# # 10 truck

# X_dict = {}
# y_dict = {}

# for i in range(0,10):
#     X_dict[i] = np.zeros(shape=(1,28,28))
#     y_dict[i] = np.zeros(shape=(1,10))

# # print(X_dict[0].shape)
# # print(y_dict[0].shape)

# for i in range(0,num_train_imgs):
#     label = y_train[i,:]
#     digit = np.nonzero(label)[0][0]
#     X_dict[digit] = np.vstack((X_dict[digit], X_train_new[i,:,:][None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


# X_dist = np.zeros(shape=(1,28,28))
# y_dist = np.zeros(shape=(1,10))

# # i is each unique class
# # dict indexing is to: a) avoid empty first vector in dict value (value is a 2D array),
# # b) only take first 100 filled vectors from the value per key
# for i in range(0,10):
#     X_dist = np.vstack((X_dist, X_dict[i][1:101,:,:]))
#     y_dist = np.vstack((y_dist, y_dict[i][1:101,:]))

# # remove first empty vector of X_dist, y_dist to generate final image and label set
# X_dist = X_dist[1:,:,:]
# y_dist = y_dist[1:,:]

# # verify array shape and presence & type of raw data of 100 imgs x 10 imgs dataset

# # print('X_dist.shape')
# # print(X_dist.shape)
# # print('y_dist.shape')
# # print(y_dist.shape)
# # for i in range(0,101):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])

# # visually verify 100 imgs by 10 classes practice set by printing

# # for i in range(899,X_dist.shape[0]):
# #     #mpl renders as pseudocolor unless cmap='gray', vmin=0, vmax=255
# #     plt.imshow(X_dist[i,:,:],cmap='gray', vmin=0, vmax=255)
# #     plt.show()



"""
Load MNIST images and prepare 1,000 images, 100 of each digit (0-9)

1,000 DOWNSAMPLED 24x24 images (for tiled model)

"""

# # load data
# # frac_samp 0.166 = 10000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=0.166,return_test=True)

# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)

# # make evenly distributed dataset of digits 0 - 9, 100 digits each

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
#     image = X_train[i,:,:]
#     resized_image = cv2.resize(image,(24,24))
#     X_dict[digit] = np.vstack((X_dict[digit], resized_image[None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


# X_dist = np.zeros(shape=(1,24,24))
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

10,000 NORMAL 28x28 images (for non-tiled model)

"""

# # load data
# # frac_samp None or 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=1,return_test=True)

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
Load MNIST images and prepare 10,000 images, 1,000 of each digit (0-9)

10,000 DOWNSAMPLED 24x24 images (for tiled model)

"""

# # load data
# # frac_samp None or 1 = 60000 images
# (X_train, y_train),(X_test,y_test) = get_keras_data(dataset='mnist',frac_samp=1,return_test=True)

# # number of initial training images
# num_imgs = y_train.shape[0]
# # print(num_imgs)

# # make evenly distributed dataset of digits 0 - 9, 100 digits each

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
#     image = X_train[i,:,:]
#     resized_image = cv2.resize(image,(24,24))
#     X_dict[digit] = np.vstack((X_dict[digit], resized_image[None,:,:]))
#     y_dict[digit] = np.vstack((y_dict[digit], label[None,:]))


# X_dist = np.zeros(shape=(1,24,24))
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

# # verify array shape and presence & type of raw data of 100 dig x 10 imgs dataset

# # print(X_dist.shape)
# # print(y_dist.shape)
# # for i in range(0,1001):
# #     print(y_dist[i])
# # for i in range(0,12):
# #     print(X_dist[i])

# # visually verify 1000 dig by 10 imgs practice set by printing

# for i in range(8999,X_dist.shape[0]):
#     plt.imshow(X_dist[i,:,:])
#     plt.show()



"""
"Out of bag" (non-training) image, and "In bag" normal , "In bag" scrambled images
"""

# out of bag image
non_training_img = np.copy(X_test[0,:,:])

# # out of bag image cifar
# non_training_img = np.copy(X_test_new[0,:,:])

# for 28x28 image
non_training_img = cv2.resize(non_training_img, (28,28))

# # for 24x24 image
# non_training_img = cv2.resize(non_training_img, (24,24))

# verify image
# plt.imshow(non_training_img, cmap='gray', vmin=0, vmax=255),plt.title('out of bag image cifar')
# plt.show()


# in bag image
training_img = np.copy(X_dist[0,:,:])
# in bag scrambled
scrambled = np.copy(X_dist[0,:,:])
scrambled = scrambled.ravel()
np.random.shuffle(scrambled)

# # for 28x28 image
scrambled = scrambled.reshape(28,28)

# # for 24x24 image
# scrambled = scrambled.reshape(24,24)

# # verify in bag normal and scrambled images
# plt.subplot(121), plt.imshow(training_img, cmap='gray', vmin=0, vmax=255),plt.title('in bag normal cifar')
# plt.subplot(122), plt.imshow(scrambled, cmap='gray', vmin=0, vmax=255),plt.title('in bag scrambled cifar')





"""
Lena loading and pre-processing
"""


lena_pw_path = 'non_mnist_images/lena_128x128_grey_prewhitened.png'
lena_zoom_path = 'non_mnist_images/lena_128x128_grey_zoomed.png'

# second arg of imread is 0 to denote reading in greyscale mode
lena_pw_read = cv2.imread(lena_pw_path,0)
lena_zoom_read = cv2.imread(lena_zoom_path,0)

# for 28x28
lena_pw = cv2.resize(lena_pw_read,(28,28))
lena_zoom = cv2.resize(lena_zoom_read,(28,28))

# # for 24x24
# lena_pw = cv2.resize(lena_pw_read,(24,24))
# lena_zoom = cv2.resize(lena_zoom_read,(24,24))


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
# plt.imshow(X_stdized[0,:,:], cmap='Greys'),plt.title('in bag standardized database')
# plt.show()

# # stdize and flatten one (the first, [0,:,:]) image from main database
# training_img_std = standardization_filter(training_img[None,:,:])
# training_img_flat = flatten_images(training_img_std)
# # #verify stdization
# training_img_sq = np.squeeze(training_img_std)
# plt.imshow(training_img_sq, cmap='Greys'),plt.title('in bag standardized')
# plt.show()

# # stdize/flatten out of bag image
# non_training_img_std = standardization_filter(non_training_img[None,:,:])
# non_training_img_flat = flatten_images(non_training_img_std)
# # #verify stdization
# non_training_img_sq = np.squeeze(non_training_img_std)
# plt.imshow(non_training_img_sq, cmap='Greys'),plt.title('out of bag standardized')
# plt.show()

# # stdize/flatten in bag scrambled image
# scrambled_std = standardization_filter(scrambled[None,:,:])
# scrambled_flat = flatten_images(scrambled_std)
# # # verify stdization
# scrambled_sq = np.squeeze(scrambled_std)
# plt.imshow(scrambled_sq, cmap='Greys'),plt.title('scrambled standardized')
# plt.show()

# # stdize/flatten lena prewhitened
# lena_pw_std = standardization_filter(lena_pw[None,:,:])
# lena_pw_flat = flatten_images(lena_pw_std)
# # # verify stdization
# lena_pw_sq = np.squeeze(lena_pw_std)
# plt.imshow(lena_pw_sq, cmap='Greys'),plt.title('lena prewhitened')
# plt.show()

# # flatten lena zoomed for pickle output
# lena_zoom_std = standardization_filter(lena_zoom[None,:,:])
# lena_zoom_flat = flatten_images(lena_zoom_std)
# # # verify stdization
# lena_zoom_sq = np.squeeze(lena_zoom_std)
# plt.imshow(lena_zoom_sq, cmap='Greys'),plt.title('lena zoom')
# plt.show()


# test out


# # pickle the flattened input images and the output vectors as a tuple
# linear_data_out = open('linear_100x10_size_24x24.pydb','wb')
# pickle.dump((X_flat, y_dist, training_img_flat, non_training_img_flat, scrambled_flat, lena_pw_flat, lena_zoom_flat), linear_data_out)
# linear_data_out.close()


# NOTE: comment out code between these NOTES when pickling the Tanh dataset


"""
Scaling and Pickling for Tanh Model
"""

# # NOTE: comment out code between these NOTES when pickling the Linear dataset


# scale main database to [-1,1] and flatten
X_tanh = rescaling_filter(X_dist, scaling_range=[-1,1])
X_flat_tanh = flatten_images(X_tanh)
# # verify tanh scaling
plt.imshow(X_tanh[0,:,:], cmap='Greys'),plt.title('in bag tanh database fashion mnist')
plt.show()

# scale in bag image to [-1,1] and flatten
training_img_tanh = rescaling_filter(training_img, scaling_range=[-1,1])
training_img_tanh_flat = flatten_images(training_img_tanh[None,:,:])
# # verify tanh scaling
plt.imshow(training_img_tanh, cmap='Greys'),plt.title('in bag tanh fashion mnist')
plt.show()

# scale out of bag image to [-1,1] and flatten
non_training_img_tanh = rescaling_filter(non_training_img, scaling_range=[-1,1])
non_training_img_tanh_flat = flatten_images(non_training_img_tanh[None,:,:])
# # verify tanh scaling
plt.imshow(non_training_img_tanh, cmap='Greys'),plt.title('out of bag tanh fashion mnist')
plt.show()

# scale in bag scrambled to [-1,1] and flatten
scrambled_tanh = rescaling_filter(scrambled, scaling_range=[-1,1])
scrambled_tanh_flat = flatten_images(scrambled_tanh[None,:,:])
# # verify tanh scaling
plt.imshow(scrambled_tanh, cmap='Greys'),plt.title('in bag scrambled tanh fashion mnist')
plt.show()


# scale lena pw to [-1,1] and flatten
lena_pw_tanh = rescaling_filter(lena_pw, scaling_range=[-1,1])
lena_pw_tanh_flat = flatten_images(lena_pw_tanh[None,:,:])
# # verify tanh scaling
plt.imshow(lena_pw_tanh, cmap='Greys'),plt.title('lena pw tanh')
plt.show()


# scale lena zoom to [-1,1] and flatten
lena_zoom_tanh = rescaling_filter(lena_zoom, scaling_range=[-1,1])
lena_zoom_tanh_flat = flatten_images(lena_zoom_tanh[None,:,:])
# # verify tanh scaling
plt.imshow(lena_zoom_tanh, cmap='Greys'),plt.title('lena zoom tanh')
plt.show()

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

`
'''
Pickle out whatever tanh dataset has been created above
'''

# # pickle the flattened input images and the output vectors as a tuple
# tanh_data_out = open('tanh_100x10_fashion_mnist.pydb', 'wb')
# pickle.dump((X_flat_tanh, y_dist, training_img_tanh_flat, non_training_img_tanh_flat, scrambled_tanh_flat, lena_pw_tanh_flat, lena_zoom_tanh_flat), tanh_data_out)
# tanh_data_out.close()


# NOTE: comment out code between these NOTES when pickling the Linear dataset
