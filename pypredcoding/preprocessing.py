 from data import get_mnist_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors, diff_of_gaussians_filter
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random
import sys.exit


""" Preprocess image data to be input into Predictive Coding Classifier """

def load_raw_imgs(data_source, num_imgs, numxpxls, numypxls):

    if data_source == "rb99":
        rb99_name_strings = ["monkey", "swan", "rose", "zebra", "forest"]
        raw_imgs = []

        # If set between 128x128 and 512x408
        if numxpxls > 128 or numypxls > 128:
            for img in range(0, num_imgs):
                path_name = "non_mnist_images/rb99_512x408/{}.png".format(rb99_name_strings[img])
                read_img = cv2.imread(path_name)
                raw_imgs.append(read_img)

        # If set smaller than 128x128
        else:
            for img in range(0, num_imgs):
                path_name = "non_mnist_images/rb99_128x128/{}.png".format(rb99_name_strings[img])
                read_img = cv2.imread(path_name)
                raw_imgs.append(read_img)

    else:
        print("load_raw_imgs(): non-rb99 image loading not yet written")

    return np.array(raw_imgs)

def convert_to_gray():
    return grayed_imgs

def apply_gaussian_mask()
    return gm_images

def apply_DoG():
    return DoG_images

def apply_tanh()
    return tanh_imgs

def cut_into_tiles(images, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):

    return tiles

def preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):

    if data_source == "rb99":
        if num_imgs != 5:
            print("rb99 has 5 images; num_imgs must == 5)

        else:
            raw_imgs = load_raw_imgs(data_source, num_imgs, numxpxls, numypxls)
            if prepro == "lifull":
                grayed_imgs = convert_to_gray(raw_imgs)
                gm_imgs = apply_gaussian_mask(grayed_imgs)
                dog_imgs = apply_DoG(gm_imgs)
                tanh_imgs = apply_tanh(dog_imgs)
                # Input
                X = tanh_imgs

            elif prepro == "grayonly":
                grayed_imgs = convert_to_gray(raw_imgs)
                # Input
                X = grayed_imgs

            elif prepro == "graytanh":
                grayed_imgs = convert_to_gray(raw_imgs)
                tanh_imgs = apply_tanh(grayed_imgs)
                # Input
                X = tanh_imgs

            else:
                print("preprocess(): prepro schema other than Li's full suite (lifull), grayed only (grayonly), and gray+tanh (graytanh) not yet written")
                exit()

            # Labels (5, 5), one hot diagonal from top left to bottom right
            y = np.eye(5)

            # If tiling desired, cut into tiles
            if tlornot == "tl":
                X = cut_into_tiles(X, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)

    elif data_source == "rao99":
        print("preprocess(): rao99 loading not yet written")
        exit()

    elif data_source == "mnist":
        print("preprocess(): mnist loading not yet written")
        exit()

    elif data_source == "cifar10":
        print("preprocess(): cifar10 loading not yet written")
        exit()
    # if data_source == "fmnist":
    else:
        print("preprocess(): fmnist loading not yet written")
        exit()

    return X, y
