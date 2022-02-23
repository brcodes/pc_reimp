#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:08:01 2022

@author: everett
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 600
from matplotlib import pyplot as plt
import pickle
from learning import *
from functools import partial
import math
# import data
import cv2
from parameters import ModelParameters
from sys import exit


""" 
general troubleshooting (save scripts here until pushing finished results)
"""
    

# """
# Tiled training case
# 30 simultaneous rU updates case
# """

# """
# variables to transfer when importing into model.py

# num_epochs -> self.p.num_epochs
# num_training_imgs -> self.p.num_training_imgs
# update_scheme -> self.p.update_scheme
# """

# # Surrogate variables

# num_epochs = 1
# numtiles = 225
# update_scheme = "rU_simultaneous"
# num_synced_rU_upds_per_img = 30

# desired_dataset = "ds.rb99_5_lifull_lin_128_128_tl_225_16_16_8_8.pydb"

# dataset_in = open(desired_dataset, "rb")
# X, y = pickle.load(dataset_in)
# dataset_in.close()

# # Li case
# # X (1125, 16, 16)
# # Y (5, 5)




# # def train(self, X, Y):
    
# # Number of training images
# num_training_images = int(X.shape[0] / numtiles)


# # Logic specific to parsing and training on a tiled X
# imgidxlo = 0
# imgidxhi = numtiles

# tiles_parsed_for_all_imgs = []

# for img in range(0, num_training_images):
    
#     tiles_of_one_img = X[imgidxlo:imgidxhi]

#     tiles_parsed_for_all_imgs.append(tiles_of_one_img)
    
#     imgidxlo += numtiles
#     imgidxhi += numtiles
    
    
# tiles_parsed_for_all_imgs = np.array(tiles_parsed_for_all_imgs, dtype=list)
    
# # Parsed X set as new X
# X = tiles_parsed_for_all_imgs


# for epoch in range(0,num_epochs):
    
#     # Shuffle order of training set input image / output vector pairs each epoch
#     N_permuted_indices = np.random.permutation(X.shape[0])
#     X_shuffled = X[N_permuted_indices]
#     Y_shuffled = Y[N_permuted_indices]
    
#     E = 0
#     C = 0
    
#     # Online classification success counter
#     num_correct = 0
    
#     # # Set learning rates at the start of each epoch
#     # k_r = self.k_r_lr(epoch)
#     # k_U = self.k_U_lr(epoch)
#     # k_o = self.k_o_lr(epoch)
    
#     k_r = 0.0005
#     k_U = 0.005
#     k_o = 0.00005
    
#     print("Epoch {}".format(epoch+1))
    
#     for tile in range(0, num_training_images):
        
#         # Copy first image into r[0]
#         single_tile = X_shuffled[image,:][:,None]
#         self.r[0] = single_image
    
    
    
    

b = np.random.rand(225, 256, 32)
a = np.random.rand(225,32,1)



# print(b)
print(b.shape)
print(a.shape)

c = np.matmul(b, a)
print(c.shape)
csqueez = c.squeeze()
print(csqueez.shape)

i = np.random.randn(225,256)

e0 = i- csqueez
print(e0.shape)
