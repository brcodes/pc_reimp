import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random
from sys import exit
import os


"""
Preprocess image data to be input into Predictive Coding Classifier
"""

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
    print("shape of images is")
    print(images.shape)
    if len(images.shape) == 4:
        grayed_imgs = []
        for img in images:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            grayed_imgs.append(gray_img)

    # If already grayscale
    else:
        print("convert_to_gray(): input images detected to be already grayscale: input matrix shape is (numimgs,xpxls,ypxls" \
        " [vs RGB input matrix shape of (numimgs, xpxls, ypxls, 3)]")
        exit()

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
        DoG_images.append(DoG_img)

    return np.array(DoG_images)

def apply_tanh(images):

    tanh_imgs = []
    for img in images:
        tanh_img = rescaling_filter(img,scaling_range=[-1,1])
        tanh_imgs.append(tanh_img)

    return np.array(tanh_imgs)

def cut_into_tiles(images, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):
    ### Cut images into tiles

    ## Tile size check
    if numtlxpxls > numxpxls:
        print("cut_into_tiles(): tile x dimension ({}) cannot be bigger than image x dimension ({})".format(numtlxpxls, numxpxls))
        exit()
    elif numtlypxls > numypxls:
        print("cut_into_tiles(): tile y dimension ({}) cannot be bigger than image y dimension ({})".format(numtlypxls, numypxls))
        exit()
    # If passes size check
    else:
        tiles_all_imgs = []

        # If only horizontal offset (will default to RB99; 3 tiles centered in image center)
        if tlyoffset == 0 and tlxoffset != 0:
            tilecols = numtiles

            # Find image "center" using first image
            center_x = int(numxpxls / 2)
            center_y = int(numypxls / 2)

            # Tile 1 (center left)
            tl1xidxlo = int(center_x - numtlxpxls / 2 - numtlxpxls)
            tl1xidxhi = int(center_x - numtlxpxls / 2)
            tl1yidxlo = int(center_y - numtlypxls / 2)
            tl1yidxhi = int(center_y + numtlypxls / 2)
            # Tile 2 (center)
            tl2xidxlo = int(center_x - numtlxpxls / 2)
            tl2xidxhi = int(center_x + numtlxpxls / 2)
            tl2yidxlo = int(center_y - numtlypxls / 2)
            tl2yidxhi = int(center_y + numtlypxls / 2)
            # Tile 3 (center right)
            tl3xidxlo = int(center_x + numtlxpxls / 2)
            tl3xidxhi = int(center_x + numtlxpxls / 2 + numtlxpxls)
            tl3yidxlo = int(center_y - numtlypxls / 2)
            tl3yidxhi = int(center_y + numtlypxls / 2)

            for img in images:
                tiles_one_img = []
                img_tl1 = img[tl1xidxlo:tl1xidxhi][tl1yidxlo:tl1yidxhi]
                img_tl2 = img[tl2xidxlo:tl2xidxhi][tl2yidxlo:tl2yidxhi]
                img_tl3 = img[tl3xidxlo:tl3xidxhi][tl3yidxlo:tl3yidxhi]
                tiles_one_img.append(img_tl1, img_tl2, img_tl3)
                tiles_all_imgs.append(tiles_one_img)

        # If only vertical offset
        elif tlyoffset != 0 and tlxoffset == 0:
            tilerows = numtiles
            print("cut_into_tiles(): vertical offset only, not yet written")
            exit()

        # If vertical and horizontal offset (RB97a: 4 tiles no overlap and Li: 225 tiles, 8px x,y overlap)
        else:
            tilecols = int(np.sqrt(numtiles))
            tilerows = tilecols

            for img in images:
                tiles_one_img = []

                tlxidxlo = 0
                tlxidxhi = numtlxpxls

                for tilecol in range(0, tilecols):

                    while tlxidxhi <= numxpxls:

                        tlyidxlo = 0
                        tlyidxhi = numtlypxls

                        for tilerow in range(0, tilerows):

                            while tlyidxhi <= numypxls:

                                tile = img[tlxidxlo:tlxidxhi][tlyidxlo:tlyidxhi]
                                tiles_one_img.append(tile)

                                # Advance cutting by one tile up the column (advance one row's height up)
                                tlyidxlo += tlyoffset
                                tlyidxhi += tlyoffset

                        tiles_all_imgs.append(tiles_one_img)

                        # After column cutting is finished, advance by one column's width to the right
                        tlxidxlo += tlxoffset
                        tlxidxhi += tlxoffset

    # Convert to numpy array
    tiles_all_imgs = np.array(tiles_all_imgs, dtype=list)

    print("size of tiles all images: {}".format(tiles_all_imgs.shape))
    print("size of tiles all images[0] (aka tiles one img, first image): {}".format(tiles_all_imgs[0].shape))
    return tiles_all_imgs

def preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):

    if data_source == "rb99":
        if num_imgs != 5:
            print("rb99 has 5 images; num_imgs must == 5")

        else:
            raw_imgs = load_raw_imgs(data_source, num_imgs, numxpxls, numypxls)
            if prepro == "lifull_lin":
                grayed_imgs = convert_to_gray(raw_imgs)
                gm_imgs = apply_gaussian_mask(grayed_imgs)
                dog_imgs = apply_DoG(gm_imgs)
                # Input
                X = dog_imgs

            elif prepro == "lifull_tanh":
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
                print("preprocess(): prepro schema other than Li's full suite (lifull_lin, lifull_tanh), grayed only (grayonly), and gray+tanh (graytanh) not yet written")
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

def dataset_find_or_create(data_source="rb99", num_imgs=5, prepro="lifull",
    numxpxls=128, numypxls=128, tlornot="tl", numtiles=225,
    numtlxpxls=16, numtlypxls=16, tlxoffset=8, tlyoffset=8):

    ### Check for dataset in local directory: if present, load; if not, create, save for later
    # Default values are Li classification set values

    desired_dataset = "ds.{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pydb".format(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)

    if os.path.exists("./" + desired_dataset):
        if __name__ == "__main__":
            dataset_in = open(desired_dataset, "rb")
            X_train, y_train = pickle.load(dataset_in)
            dataset_in.close()

            print("\n" + "I. Dataset " + desired_dataset + " successfully loaded from local dir" + "\n")

        # I.e. if __name__ == "__preprocessing__", or we are in some other script
        else:
            print("Desired dataset " + desired_dataset + " already present in local dir: would you like to overwrite it? (y/n)")
            ans = input()
            # For overwrite
            if ans == "y":
                # Create dataset per specifications
                X_train, y_train = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
                dataset_out = open(desired_dataset, "wb")
                pickle.dump((X_train, y_train), dataset_out)
                dataset_out.close()

            else:
                print("Quitting dataset creation..." + "\n")
                exit()

    else:
        X_train, y_train = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
        dataset_out = open(desired_dataset, "wb")
        pickle.dump((X_train, y_train), dataset_out)
        dataset_out.close()

    return X_train, y_train


####

### If you want to preprocess and pickle a dataset outside of a main.py model-training operation, or just overwrite an old one, do it here.

### Set parameters of datset to create

## Data source
data_source = "rb99"
# data_source = "rao99"
# data_source = "rb97a"
# data_source = "mnist"

## Number of images
num_imgs = 5
# num_imgs = 10
# num_imgs = 100
# num_imgs = 1000
# num_imgs = 10000
# num_imgs = 600000

## Preprocessing scheme
prepro = "lifull_lin"
# prepro = "lifull_tanh"
# prepro = "grayonly"
# prepro = "graytanh"

## Image x,y dimensions
# numxpxls, numypxls = 28, 28
# numxpxls, numypxls = 38, 38
# numxpxls, numypxls = 48, 48
# numxpxls, numypxls = 68, 68
numxpxls, numypxls = 128, 128
# numxpxls, numypxls = 512, 408
# numxpxls, numypxls = 512, 512

## Tiled or not
tlornot = "tl"
# tlornot = "ntl"

## Number of tiles
# numtiles = 0
# numtiles = 3
numtiles = 225

## Tile x,y dimensions
# numtlxpxls, numtlypxls = 0, 0
# numtlxpxls, numtlypxls = 15, 15
numtlxpxls, numtlypxls = 16, 16
# numtlxpxls, numtlypxls = 12, 24

## Tile x,y offset
# tlxoffset, tlyoffset = 0, 0
# tlxoffset, tlyoffset = 5, 0
# tlxoffset, tlyoffset = 6, 0
tlxoffset, tlyoffset = 8, 8


X, y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
