from data import get_mnist_data,flatten_images,standardization_filter,rescaling_filter, inflate_vectors
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
                if numxpxls == 512 and numypxls == 408:
                    raw_imgs.append(read_img)
                else:
                    resized_img = cv2.resize(read_img, (numxpxls, numypxls))
                    raw_imgs.append(resized_img)

        # If set smaller than 128x128
        else:
            for img in range(0, num_imgs):
                path_name = "non_mnist_images/rb99_128x128/{}.png".format(rb99_name_strings[img])
                read_img = cv2.imread(path_name)
                if numxpxls == 128 and numypxls == 128:
                    raw_imgs.append(read_img)
                else:
                    resized_img = cv2.resize(read_img, (numxpxls, numypxls))
                    raw_imgs.append(resized_img)

    else:
        print("load_raw_imgs(): non-rb99 image loading not yet written")

    return np.array(raw_imgs)

def convert_to_gray(images):
    ### Only supports conversion from RGB (RB99, Rao99, CIFAR-10); MNIST, FMNIST already grayscale
    # Check RGB status here
        if len(images.shape) == 3:
        grayed_imgs = []
        for img in images:
            gray_img = cv2.cvtColor(img)
            grayed_imgs.append(gray_img)

        # If already grayscale
        else:
            print("convert_to_gray(): input images detected to be already grayscale; input matrix shape (n,n)")

    return np.array(grayed_imgs)

def apply_gaussian_mask(images, sigma=1.0, numxpxls=128, numypxls=128):
    ### Apply gaussian mask (Li default parameters)
    mu = 0.0
    x, y = np.meshgrid(np.linspace(-1,1,numxpxls), np.linspace(-1,1,numypxls))
    d = np.sqrt(x**2+y**2)
    g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
    mask = g / np.max(g)

    gm_images = []
    for img in images:
        mask_img = img * mask
        gm_images.append(mask_img)

    return np.array(gm_images)

def apply_DoG(images, ksize=(5,5), sigma1=1.3, sigma2=2.6):
    ### Apply Difference of Gaussians filter for image edge detection (Li default parameters)

    DoG_images = []
    for img in images:
        g1 = cv2.GaussianBlur(img, ksize, sigma1)
        g2 = cv2.GaussianBlur(img, ksize, sigma2)
        DoG_img = g1 - g2
        DoG_images.append(DoG)

    return np.array(DoG_images)

def apply_tanh(images):

    tanh_imgs = []
    for img in images:
        tanh_img = rescaling_filter(img,scaling_range=[-1,1])
        tanh_imgs.append(tanh_img)

    return np.array(tanh_imgs)

def cut_into_tiles(images, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):
    ### Cut images into tiles
    # images must be

    ## Tile size check
    if numtlxpxls > numxpxls:
        print("cut_into_tiles(): tile x dimension ({}) cannot be bigger than image x dimension ({})".format(numxpxls, numtlxpxls))
        exit()
    elif numtlypxls > numypxls:
        print("cut_into_tiles(): tile y dimension ({}) cannot be bigger than image y dimension ({})".format(numypxls, numtlypxls))
        exit()
    # If passes size check
    else:
        tiles_all_imgs = []

        # If only horizontal offset (RB99)
        if tlyoffset == 0 and tlxoffset != 0:
            tilecols == numtiles
            # Find image "center" using first image
            center_x = int(numxpxls / 2)
            center_y = int(numypxls / 2)
            for img in images:


        # If only vertical offset
        elif tlyoffset != 0 and tlxoffset == 0:
            tilerows = numtiles
            print("cut_into_tiles(): vertical offset only, not yet written")
            exit()

        # If vertical and horizontal offset (RB97a and Li)
        else:
            for img in images:
                tiles_one_img = []
                tilecols = np.sqrt(numtiles)
                for tilecol in range(0, tilecols):

                tiles_all_imgs.append(tiles_one_img)

    return tiles_all_imgs

def preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):

    if data_source == "rb99":
        if num_imgs != 5:
            print("rb99 has 5 images; num_imgs must == 5")

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
