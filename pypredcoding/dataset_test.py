from matplotlib import pyplot as plt
import cv2
import os.path
import pickle

"""
Grab and print a dataset of your choice to verify its contents
"""

### Set parameters of datset to import

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


desired_dataset = "ds.{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pydb".format(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)

if os.path.exists("./" + desired_dataset):

    print("\n" + "Desired dataset " + desired_dataset + "exists in local dir.")
    dataset_in = open(desired_dataset, "rb")
    X, y = pickle.load(dataset_in)
    dataset_in.close()
    print("Desired dataset " + desired_dataset + "imported. Printing components:")

    #X should be sized (num_imgs, numxpxls, numypxls) if gray
    #X should be sized (num_imgs, numxpxls, numypxls, 3) if RGB or BGR color
    #y should be sized (num_imgs, num_classes), where num_classes is gleaned from data_source as either 5 (RB99, Rao99, RB97a) or 10 (MNIST, Fashion MNIST, CIFAR10)

    if data_source == "rb99" or data_source == "rb97a" or data_source == "rao99":
        num_classes = 5

    elif data_source == "mnist" or data_source == "fmnist" or data_source == "cifar10":
        num_classes = 10


    # Gray case
    if len(X.shape) == 3:
        "Grayscale images detected"

        # Whole image case
        if tlornot == "ntl":
            img_num = 1
            for img in X:
                plt.imshow(img)
                plt.title("{}".format(desired_dataset) + "\n" + "image {}".format(img_num))
                plt.show()
                img_num += 1

        # Tiled image case
        elif tlornot == "tl":
            
            imgidxlo = 0
            imgidxhi = numtiles

            for img in range(0, num_imgs):

                img_in_x = X[imgidxlo:imgidxhi]

                tl_num = 1
                for tl in img_in_x:
                    plt.imshow(tl)
                    plt.title("{}".format(desired_dataset) + "\n" + "image {}".format(img+1) + "tile {}".format(tl_num))
                    plt.show()
                    tl_num += 1

                imgidxlo += numtiles
                imgidxhi += numtiles

            

    # Color case
    elif len(X.shape) == 4:
        "Color images detected"

        # Whole image case
        if tlornot == "ntl":
            img_num = 1
            for img in X:
                plt.imshow(img)
                plt.title("{}".format(desired_dataset) + "\n" + "image {}".format(img_num))
                plt.show()
                img_num += 1




else:

    print("Desired dataset " + desired_dataset + "does not exist in local dir: please create using data.py or main.py")