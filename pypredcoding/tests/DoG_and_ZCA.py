#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:12:57 2020

@author: everett
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.datasets import mnist
from numpy import ceil


"""
Data Setup
"""
# from KB

# Get data from data.py using frac_samp feature
frac_samp = 0.00008

# read from Keras
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# check initial shape
print(X_train.shape)

# prune data
if frac_samp is not None:
    n_train = int(ceil(frac_samp*X_train.shape[0]))
else:
    n_train = X_train.shape[0]

# X_train now contains n_train number of images
X_train = X_train[:n_train,:,:]

#check pruned shape
print(X_train.shape)


"""
Perform ZCA Standardization per-pixel
"""
# from KDnuggets

# flatten images
X_train_z = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
print(X_train_z.shape)

# check shape of example single flat image pre-norm
Sngl_img = X_train_z[3,:].reshape(1,784)
print(Sngl_img.shape)

# normalize
X_norm = X_train_z / 255
print(X_norm.shape[0])

# check shape of example single flat image post-norm
Sngl_img = X_norm[3,:].reshape(1,784)
print(Sngl_img.shape)
print(Sngl_img.min())
print(Sngl_img.max())

# subtract mean pixel value from each pixel in each image
print(X_norm.mean(axis=0).shape)
X_norm = X_norm - X_norm.mean(axis=0)

# create covariance matrix
cov = np.cov(X_norm, rowvar=False)
print(cov.shape)

# single value decomposition of covariance matrix
U,S,V = np.linalg.svd(cov)
print("U and S shapes:")
print(U.shape, S.shape)

# ZCA
epsilon = 0.1
X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm.T).T
print(X_ZCA.shape)

# rescale (what does this do? - noticed no visible difference w/wo)
X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
print(X_ZCA_rescaled.min())
print(X_ZCA_rescaled.max())
print(X_ZCA_rescaled.shape)

# inflate back to 2d square
X_ZCA_infl = X_ZCA_rescaled.reshape(n_train,28,28)
print(X_ZCA_infl.shape)


"""
Compare DoG to Non-DoG for ZCA and Non-ZCA
"""

ksize = (5,5)
sigma1 = 1.3
sigma2 = 2.6

num_images = n_train

for i in range(0, num_images):
    image = X_train[i,:,:]
    g1 = cv2.GaussianBlur(image, ksize, sigma1)
    g2 = cv2.GaussianBlur(image, ksize, sigma2)
    dog_img = g2 - g1
    plt.subplot(121),plt.imshow(image, cmap='Greys'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122),plt.imshow(dog_img, cmap='Greys'),plt.title('DoG')
    plt.xticks([]), plt.yticks([])
    plt.show()

for i in range(0, num_images):
    image_ZCA = X_ZCA_infl[i,:,:]
    g1 = cv2.GaussianBlur(image_ZCA, ksize, sigma1)
    g2 = cv2.GaussianBlur(image_ZCA, ksize, sigma2)
    dog_img = g2 - g1
    plt.subplot(121),plt.imshow(image_ZCA, cmap='Greys'),plt.title('ZCA')
    plt.xticks([]), plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122),plt.imshow(dog_img, cmap='Greys'),plt.title('ZCA DoG')
    plt.xticks([]), plt.yticks([])
    plt.show()


















# to load images from pc_project dir ("images_rao")
# image = cv2.imread('image{}.png'.format(i))
