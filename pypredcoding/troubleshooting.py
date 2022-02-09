#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:08:01 2022

@author: everett
"""
import numpy as np

from matplotlib import pyplot as plt

# if 1 == 0:
#     print(1)
    
# elif 2 == 2:
#     print(2)
    
    
# if 1 == 0:
#     print(1)
    
# elif 2 == 2:
#     print(2)
    
    
    
# X = np.zeros((1125,16,16))

# # for img in X:
# #     print(img)
    
# desired_dataset = 'ds.rb99_5_lifull_lin_128_128_tl_225_16_16_8_8.pydb'

# num_imgs = 5
# numtiles = 225
    
    
# # Gray case
# if len(X.shape) == 3:
#     "Grayscale images detected"
    
#     imgidxlo = 0
#     imgidxhi = numtiles
    
#     for img in range(0, num_imgs):
                
#         img_in_x = X[imgidxlo:imgidxhi]

#         tl_num = 1
#         for tl in img_in_x:
#             plt.imshow(tl)
#             plt.title("{}".format(desired_dataset) + "\n" + "image {} ".format(img+1) + "tile {}".format(tl_num))
#             plt.show()
#             tl_num += 1

#         imgidxlo += numtiles
#         imgidxhi += numtiles


a = np.array([[1,2],[3,4]])
print(a)
print(a.shape)
b = np.array([[5,6],[7,8]])
print(b)
print(b.shape)

c = np.concatenate((a,b), axis=1)
print(c)
print(c.shape)

list = [1,2,3,4]

l = list[0:0]
print(l)
