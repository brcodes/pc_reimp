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


""" 
dataset_test troubleshooting
"""
    
    
    
# X = np.ones((225,16,16))

# for img in X:
#     print(img)
    
desired_dataset = 'ds.rb99_5_lifull_lin_128_128_tl_225_16_16_8_8.pydb'

dataset_in = open(desired_dataset, "rb")
X, y = pickle.load(dataset_in)
dataset_in.close()


xshape = X.shape
x0shape = X[0].shape

    

num_imgs = 5
numtiles = 225
numrows = 15
numcols = 15
tlxoffset, tlyoffset = 8, 8
numtlxpxls, numtlypxls = 16, 16
    


imgidxlo = 0
imgidxhi = numtiles

tiles_of_all_images = []

# Tile parsing and plotting (singles) loop

for img in range(0, num_imgs):
            
    img_in_x = X[imgidxlo:imgidxhi]
    
    tiles_of_single_img = []
    
    tl_num = 1
    
    for tl in img_in_x:
        
        tiles_of_single_img.append(tl)
        
        # Single tile plotting
        
        tl = np.array(tl).astype(float)

        # plt.imshow(tl, cmap="gray")
        # plt.title("{}".format(desired_dataset) + "\n" + "image {} ".format(img+1) + "tile {}".format(tl_num))
        # plt.show()
        
        tl_num += 1
        
    tiles_of_all_images.append(tiles_of_single_img)

    imgidxlo += numtiles
    imgidxhi += numtiles
    
# Tile stacking and plotting (stacked collage) loop

tiles_of_all_images = np.array(tiles_of_all_images, dtype=list)

for img in range(0, num_imgs):
    
    # In Li case, this is (225,16,16) but generally is (numtiles, tlxpxls, tlypxls)
    tiles_of_single_img = tiles_of_all_images[img]
    
    rowidxlo = 0
    rowidxhi = numcols
    
    vstackedrows = np.zeros([numtlypxls,numtlxpxls*numcols])
    
    for row in range(0,numrows):
        
        onerow = tiles_of_single_img[rowidxlo:rowidxhi,:,:]
        
        # Initiate left-to-right stacking, completing a full row
        hstackedrow = np.array(onerow[0])[:,:]
        
        for col in range(1, numcols):
            nonfirsttileinrow = np.array(onerow[col])[:,:]
            hstackedrow = np.concatenate((hstackedrow, nonfirsttileinrow), axis=1)
            
        # With collaged row complete, stack rows for full collage
        vstackedrows = np.vstack([vstackedrows, hstackedrow])
        
        rowidxlo += numcols
        rowidxhi += numcols
         
    # Remove collage row of zeros used to initiate collage, convert to float64
    stackedtilecollage = vstackedrows[numtlypxls:,:].astype(float)
    
    # Plot
    plt.imshow(stackedtilecollage, cmap="gray")
    plt.title("{} img {}".format(desired_dataset, img+1) + "\n" + 
              "{}-tile collage ({}x{}tls); each {}x{}pxls " \
              "with x, y offsets {},{}".format(numtiles, numrows, numcols, numtlxpxls, numtlypxls, tlxoffset, tlyoffset))
    
    plt.show()
        
# fig, ax_mat = plt.subplots(numrows, numcols)

# for i, ax_row in enumerate(ax_mat):
#     for j, axes in enumerate(ax_row):
#         axes.set_yticklabels([])
#         axes.set_xticklabels([])
        
# plt.show()

# a = np.array([[1,2],[3,4]])
# print(a)
# print(a.shape)
# b = np.array([[5,6],[7,8]])
# print(b)
# print(b.shape)

# c = np.concatenate((a,b), axis=1)
# print(c)
# print(c.shape)

# list = [1,2,3,4]

# l = list[0:0]
# print(l)
