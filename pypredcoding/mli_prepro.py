#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:33:58 2021

@author: everett
"""

from data import get_mnist_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors, diff_of_gaussians_filter
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random

#Monica's gaussian mask function exactly as written in sPCSWR_linear commit

def create_gauss_mask(sigma=0.4, width=16, height=16):
    """ Create gaussian mask. """
    mu = 0.0
    x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    d = np.sqrt(x**2+y**2)
    g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
    mask = g / np.max(g)
    return mask

def apply_DoG_filter(image, ksize=(5,5), sigma1=1.3, sigma2=2.6):
       """
       Apply difference of gaussian (DoG) filter detect edge of the image.
       """
       g1 = cv2.GaussianBlur(image, ksize, sigma1)
       g2 = cv2.GaussianBlur(image, ksize, sigma2)
       return g1 - g2


#Load high-res 128x128 R&B '99 images data from local path

rb_nat_monkey_path = 'non_mnist_images/images_rao_nature/monkey.png'
rb_nat_swan_path = 'non_mnist_images/images_rao_nature/swan.png'
rb_nat_rose_path = 'non_mnist_images/images_rao_nature/rose.png'
rb_nat_zebra_path = 'non_mnist_images/images_rao_nature/zebra.png'
rb_nat_forest_path = 'non_mnist_images/images_rao_nature/forest.png'

rb_nat_monkey_read = cv2.imread(rb_nat_monkey_path)
rb_nat_swan_read = cv2.imread(rb_nat_swan_path) 
rb_nat_rose_read = cv2.imread(rb_nat_rose_path)
rb_nat_zebra_read = cv2.imread(rb_nat_zebra_path)
rb_nat_forest_read = cv2.imread(rb_nat_forest_path)

#Plot in original form

# plt.imshow(np.squeeze(rb_nat_monkey_read), cmap='gray'),plt.title('r&b nature monkey 128x128')
# plt.show()

# plt.imshow(np.squeeze(rb_nat_swan_read), cmap='gray'),plt.title('r&b nature swan 128x128')
# plt.show()

# plt.imshow(np.squeeze(rb_nat_rose_read), cmap='gray'),plt.title('r&b nature rose 128x128')
# plt.show()

# plt.imshow(np.squeeze(rb_nat_zebra_read), cmap='gray'),plt.title('r&b nature zebra 128x128')
# plt.show()

# plt.imshow(np.squeeze(rb_nat_forest_read), cmap='gray'),plt.title('r&b nature forest 128x128')
# plt.show()



#Turn things "offically" gray (from RGB) and plot
monkey_gray = cv2.cvtColor(rb_nat_monkey_read, cv2.COLOR_RGB2GRAY)
swan_gray = cv2.cvtColor(rb_nat_swan_read, cv2.COLOR_RGB2GRAY)
rose_gray = cv2.cvtColor(rb_nat_rose_read, cv2.COLOR_RGB2GRAY)
zebra_gray = cv2.cvtColor(rb_nat_zebra_read, cv2.COLOR_RGB2GRAY)
forest_gray = cv2.cvtColor(rb_nat_forest_read, cv2.COLOR_RGB2GRAY)

#Plot grayed images
# plt.imshow(monkey_gray, cmap='gray'),plt.title('r&b nature monkey 128x128 RGB2GRAY')
# plt.show()
# plt.imshow(swan_gray, cmap='gray'),plt.title('r&b nature swan 128x128 RGB2GRAY')
# plt.show()
# plt.imshow(rose_gray, cmap='gray'),plt.title('r&b nature rose 128x128 RGB2GRAY')
# plt.show()
# plt.imshow(zebra_gray, cmap='gray'),plt.title('r&b nature zebra 128x128 RGB2GRAY')
# plt.show()
# plt.imshow(forest_gray, cmap='gray'),plt.title('r&b nature forest 128x128 RGB2GRAY')
# plt.show()

#Apply DoG
monkey_dog = apply_DoG_filter(monkey_gray)
swan_dog = apply_DoG_filter(swan_gray)
rose_dog = apply_DoG_filter(rose_gray)
zebra_dog = apply_DoG_filter(zebra_gray)
forest_dog = apply_DoG_filter(forest_gray)

# #Plot DoG
# plt.imshow(monkey_dog, cmap='gray'),plt.title('monkey 128x128 gray DoG')
# plt.show()
# plt.imshow(swan_dog, cmap='gray'),plt.title('swan 128x128 gray DoG')
# plt.show()
# plt.imshow(rose_dog, cmap='gray'),plt.title('rose 128x128 gray DoG')
# plt.show()
# plt.imshow(zebra_dog, cmap='gray'),plt.title('zebra 128x128 gray DoG')
# plt.show()
# plt.imshow(forest_dog, cmap='gray'),plt.title('forest 128x128 gray DoG')
# plt.show()


#Tanh rescale and flatten Non-DoG 128x128 Images
monkey_tanh = rescaling_filter(monkey_gray, scaling_range=[-1,1])
monkey_tanh_flat = flatten_images(monkey_tanh[None,:,:])

swan_tanh = rescaling_filter(swan_gray, scaling_range=[-1,1])
swan_tanh_flat = flatten_images(swan_tanh[None,:,:])

rose_tanh = rescaling_filter(rose_gray, scaling_range=[-1,1])
rose_tanh_flat = flatten_images(rose_tanh[None,:,:])

zebra_tanh = rescaling_filter(zebra_gray, scaling_range=[-1,1])
zebra_tanh_flat = flatten_images(zebra_tanh[None,:,:])

forest_tanh = rescaling_filter(forest_gray, scaling_range=[-1,1])
forest_tanh_flat = flatten_images(forest_tanh[None,:,:])


# #Plot tanh scaled images
# plt.imshow(np.squeeze(monkey_tanh), cmap='gray'),plt.title('monkey 128x128 gray tanh')
# plt.show()
# plt.imshow(np.squeeze(swan_tanh), cmap='gray'),plt.title('swan 128x128 gray tanh')
# plt.show()
# plt.imshow(np.squeeze(rose_tanh), cmap='gray'),plt.title('rose 128x128 gray tanh')
# plt.show()
# plt.imshow(np.squeeze(zebra_tanh), cmap='gray'),plt.title('zebra 128x128 gray tanh')
# plt.show()
# plt.imshow(np.squeeze(forest_tanh), cmap='gray'),plt.title('forest 128x128 gray tanh')
# plt.show()


#Tanh rescale and flatten DoG 128x128 Images

monkey_dog_tanh = rescaling_filter(monkey_dog, scaling_range=[-1,1])
monkey_dog_tanh_flat = flatten_images(monkey_dog_tanh[None,:,:])

swan_dog_tanh = rescaling_filter(swan_dog, scaling_range=[-1,1])
swan_dog_tanh_flat = flatten_images(swan_dog_tanh[None,:,:])

rose_dog_tanh = rescaling_filter(rose_dog, scaling_range=[-1,1])
rose_dog_tanh_flat = flatten_images(rose_dog_tanh[None,:,:])

zebra_dog_tanh = rescaling_filter(zebra_dog, scaling_range=[-1,1])
zebra_dog_tanh_flat = flatten_images(zebra_dog_tanh[None,:,:])

forest_dog_tanh = rescaling_filter(forest_dog, scaling_range=[-1,1])
forest_dog_tanh_flat = flatten_images(forest_dog_tanh[None,:,:])


# #Plot tanh scaled images
# plt.imshow(np.squeeze(monkey_dog_tanh), cmap='gray'),plt.title('monkey 128x128 gray DoG tanh')
# plt.show()
# plt.imshow(np.squeeze(swan_dog_tanh), cmap='gray'),plt.title('swan 128x128 gray DoG tanh')
# plt.show()
# plt.imshow(np.squeeze(rose_dog_tanh), cmap='gray'),plt.title('rose 128x128 gray DoG tanh')
# plt.show()
# plt.imshow(np.squeeze(zebra_dog_tanh), cmap='gray'),plt.title('zebra 128x128 gray DoG anh')
# plt.show()
# plt.imshow(np.squeeze(forest_dog_tanh), cmap='gray'),plt.title('forest 128x128 gray DoG tanh')
# plt.show()


#Verify that Tanh rescaling worked (that numbers are in [-1,1])
# print("size of monkey_dog_tanh")
# print(monkey_dog_tanh.shape)

# print("min(monkey_dog_tanh[0])")
# print(min(monkey_dog_tanh[0]))

# print("max(monkey_dog_tanh[0])")
# print(max(monkey_dog_tanh[0]))

# print("min(monkey_tanh[0])")
# print(min(monkey_tanh[0]))

# print("max(monkey_tanh[0])")
# print(max(monkey_tanh[0]))

# print("min(monkey_dog_tanh[1])")
# print(min(monkey_dog_tanh[1]))

# print("max(monkey_dog_tanh[1])")
# print(max(monkey_dog_tanh[1]))

# print("min(monkey_tanh[1])")
# print(min(monkey_tanh[1]))

# print("max(monkey_tanh[1])")
# print(max(monkey_tanh[1]))






nature = [monkey_tanh_flat,swan_tanh_flat,rose_tanh_flat,zebra_tanh_flat,forest_tanh_flat]
nature_dog = [monkey_dog_tanh_flat,swan_dog_tanh_flat,rose_dog_tanh_flat,zebra_dog_tanh_flat,forest_dog_tanh_flat]


combined_nat_imgs_vec = np.zeros(shape=(1,16384))
combined_nat_dog_imgs_vec = np.zeros(shape=(1,16384))
combined_labels_vec = np.zeros(shape=(1,5))



for i in range(0,5):
    #image parsing and stacking
    nat_img = nature[i]
    nat_dog_img = nature_dog[i]
    reshaped_nat = nat_img.reshape(1,16384)
    reshaped_nat_dog = nat_dog_img.reshape(1,16384)
    combined_nat_imgs_vec = np.vstack((combined_nat_imgs_vec, reshaped_nat))
    combined_nat_dog_imgs_vec = np.vstack((combined_nat_dog_imgs_vec, reshaped_nat_dog))
    
    #label creation and stacking
    label = np.zeros(shape=(1,5))
    label[:,i] = 1
    combined_labels_vec = np.vstack((combined_labels_vec,label))
    
print('final shape of combined nat imgs vec is {}'.format(combined_nat_imgs_vec.shape))
print('final shape of combined nat dog imgs vec is {}'.format(combined_nat_dog_imgs_vec.shape))
print('final shape of combined labels vec is {}'.format(combined_labels_vec.shape))
print('final combined labels vec is')
print(combined_labels_vec)
    
    
nat_imgs_vec = combined_nat_imgs_vec[1:6]
nat_dog_imgs_vec = combined_nat_dog_imgs_vec[1:6]
labels_vec = combined_labels_vec[1:6]
    
print('final shape of nat imgs vec is {}'.format(nat_imgs_vec.shape))
print('final shape of nat dog imgs vec is {}'.format(nat_dog_imgs_vec.shape))
print('final shape of labels vec is {}'.format(labels_vec.shape))
print('final combined labels vec is')
print(labels_vec)
    



#Pickle the flattened input images and the label vectors as a tuple
tanh_data_out = open('rao_ballard_nature_128x128_tanh.pydb', 'wb')
pickle.dump((nat_imgs_vec,labels_vec), tanh_data_out)
tanh_data_out.close()

tanh_data_out = open('rao_ballard_nature_128x128_DoG_tanh.pydb', 'wb')
pickle.dump((nat_dog_imgs_vec,labels_vec), tanh_data_out)
tanh_data_out.close()





















#Stuff from MLI's classification.ipynb: the reason the printed reconstruction plots are ~ 500x500 pixels (cv2 resize)
#Not sure what the norm-minmax is doing when used just for plotting, and not for dataset preprocessing...

# filtered_monkey = cv2.resize(monkey_dog, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
# filtered_monkey = cv2.normalize(src=filtered_monkey, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# plt.figure(figsize=(9, 3), dpi=80)

# plt.imshow(filtered_monkey, cmap='gray'),plt.title('monkey 128x128 gray DoG cv2resize,norm')
# plt.show()

