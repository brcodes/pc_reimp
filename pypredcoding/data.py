from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
import random
from sys import exit
import os

"""
Manipulate and preprocess image data to be input into Predictive Coding Classifier
"""


def get_mnist_data(frac_samp=None,return_test=False):
    '''
    Returns MNIST training examples (and potentially test) data.  Images (X_train and
    X_test) are returned in an array of n_samples x 28 x 28, and target patterns
    (y_train and y_test) are n_samples x num_classes-length one hot vectors.

    keras's mnist load returns 60,000 training samples and 10,000 test samples.
    Set frac_samp to a number in [0,1] to reduce the proportion of samples
    returned.
    '''
    # read from Keras
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    # conversion of target patterns to one-hot vectors
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # prune data
    if frac_samp is not None:
        n_train = int(np.ceil(frac_samp*X_train.shape[0]))
        n_test = int(np.ceil(frac_samp*X_test.shape[0]))
    else:
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
    # send it all back
    if return_test:
        return (X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64')),(X_test[:n_test,:,:].astype('float64'),y_test[:n_test,:].astype('float64'))
    return X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64')


def flatten_images(image_array):
    '''
    Accepts an array of N x dim_x x dim_y images (N images each of dim_x x dim_y size)
    and returns a 2D array of flattened images

    This function will fail on a single image unless you give it an empty first
    axis (np.newaxis,:,:).
    '''
    N = image_array.shape[0]
    dim_x = image_array.shape[1]
    dim_y = image_array.shape[2]
    return image_array.reshape(N,dim_x*dim_y)


def inflate_vectors(vector_array,shape_2d=None):
    '''
    Accepts an array of N x s flattened images (vectors) and returns an array of
    N x shape_2d[0] x shape_2d[1] images.  If shape_2d is none, images are
    assumed to be square.

    This function will fail on a single input vector unless you give it an empty
    first axis (np.newaxis,:).
    '''
    N = vector_array.shape[0]
    if shape_2d == None:
        side = int(np.sqrt(vector_array.shape[1]))
        return vector_array.reshape(N, side, side)
    else:
        height = shape_2d[0]
        width = shape_2d[1]
        return vector_array.reshape(N, height, width)


def rescaling_filter(vector_array,scaling_range=[0,1]):
    '''
    Though applied in this module to accept an array of N x s flattened images (vectors)
    and return an array of the same size, this function will accept and return an
    array of any size.

    Scales the input image range to be in [a,b]
    '''
    rescaled_array = scaling_range[0] + (scaling_range[1]-scaling_range[0])*((vector_array - vector_array.min()) / (vector_array.max() - vector_array.min()))
    return rescaled_array


def diff_of_gaussians_filter(image_array, kern_size=(5,5), sigma1=1.3, sigma2=2.6):
    '''
    Accepts an array of N x dim_x x dim_y images (N images each of dim_x x dim_y size),
    and returns an array of the same size.

    Kernel size (kern_size) must be a 2-int tuple, (x,y), where x = kernel width and
    y = kernel height in number of pixels. Default to kernel size used in Monica Li's dataset.py.

    Requires two standard deviation parameters (sigma1 and sigma2), where
    sigma2 (g2) > sigma1 (g1). These parameters facilitate subtraction of the original image
    from a less blurred version of the image. Default to values found in Monica Li's dataset.py.

    This function will fail on a single image unless you give it an empty first
    axis (np.newaxis,:,:).
    '''
    # error notice
    if sigma1 >= sigma2:
        print("Error [DoG]: sigma2 must be greater than sigma1")
        return
    # apply filter to each image
    filtered_array = np.zeros_like(image_array)
    for i in range(0, image_array.shape[0]):
        image = image_array[i,:,:]
        g1 = cv2.GaussianBlur(image, kern_size, sigma1)
        g2 = cv2.GaussianBlur(image, kern_size, sigma2)
        filtered_array[i,:,:] = g1 - g2
    return filtered_array


def standardization_filter(image_array):
    '''
    Accepts an array of N x dim_x x dim_y images (N images each of dim_x x dim_y size),
    and returns an array of the same size.

    Standardizes a sample image (1D vector) by subtracting its mean pixel value and
    then dividing by the standard deviation of the pixel values. (i.e. Z-score)

    This function will fail on a single image unless you give it an empty first
    axis (np.newaxis,:,:).
    '''
    whitened = np.zeros_like(image_array)
    for i in range(0,image_array.shape[0]):
        image = image_array[i,:,:]
        whitened[i,:,:] = (image - image.mean())/image.std()
    return whitened


def zca_filter(image_array, epsilon=0.1):
    '''
    Accepts an array of N x dim_x x dim_y images (N images each of dim_x x dim_y size),
    and returns an array of the same size.

    Epsilon is a whitening coefficient. (Sudeep 2016 uses 0.1)

    This function will fail on a single image unless you give it an empty first
    axis (np.newaxis,:,:).
    '''
    zca_imgs = np.zeros_like(image_array)
    # flatten and normalize
    flattened_images = flatten_images(image_array)
    images_norm = flattened_images / 255
    # subtract mean pixel value from each pixel in each image
    images_norm = images_norm - images_norm.mean(axis=0)
    # create covariance matrix
    cov = np.cov(images_norm, rowvar=False)
    # single value decomposition of covariance matrix
    U,S,V = np.linalg.svd(cov)
    # perform ZCA
    images_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(images_norm.T).T
    # inflate
    zca_imgs = inflate_vectors(images_ZCA)
    return zca_imgs

def cut(image_array, tile_offset, flat=True):
    '''
    Designed to take a 24x24 image array (an MNIST image downsampled from native 28x28) and cut it
    into three (rectangle) tiles whose size and amount of overlap depend on their horizontal (L -> R)
    offset from the previous tile.

    For 24x24 images, tile_offset can range from 0 (all tiles are 24x24 and fully overlap)
    to 8 (all tiles are 8x24 and do not overlap at all). tile_offset = 6 is the only value where the offset = overlap (6)

    returns a tuple.
    '''

    if flat == True:

        image_width = image_array.shape[1]

        t1_index1 = 0
        t1_index2 = image_width - (tile_offset*2)

        # take all rows (:) within some horizontal slice
        tile1 = image_array[:,t1_index1:t1_index2]
        tile1flat = flatten_images(tile1[None,:,:])

        t2_index1 = t1_index1 + tile_offset
        t2_index2 = image_width - tile_offset

        tile2 = image_array[:,t2_index1:t2_index2]
        tile2flat = flatten_images(tile2[None,:,:])

        t3_index1 = t2_index1 + tile_offset
        t3_index2 = image_width

        tile3 = image_array[:,t3_index1:t3_index2]
        tile3flat = flatten_images(tile3[None,:,:])

        return (tile1flat, tile2flat, tile3flat)

    elif flat == False:

        image_width = image_array.shape[1]

        t1_index1 = 0
        t1_index2 = image_width - (tile_offset*2)

        # take all rows (:) within some horizontal slice
        tile1 = image_array[:,t1_index1:t1_index2]

        t2_index1 = t1_index1 + tile_offset
        t2_index2 = image_width - tile_offset

        tile2 = image_array[:,t2_index1:t2_index2]

        t3_index1 = t2_index1 + tile_offset
        t3_index2 = image_width

        tile3 = image_array[:,t3_index1:t3_index2]

        return (tile1, tile2, tile3)

    else:

        print('flat must = True or False bool')
        return


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
            
            tiles_all_imgs = []
            
            # Initiate image counter for plot title
            img_num = 1

            for img in images:

                # Start at the top left corner of the image
                # Traversing across a row of tiles by tlxoffset each iteration
                # Then down by tlyoffset once row of tiles cut and stored
                
                rowidxlo = 0
                rowidxhi = numtlypxls
                
                # Initiate cut tile counter for plot title
                cut_tile_num = 1
                
                for row in range(0,tilerows):
                    # Set row vertical dimensions
                    onerow = img[rowidxlo:rowidxhi]
                    
                    print("onerow size is {}".format(onerow.shape))
                    
                    tlidxlo = 0
                    tlidxhi = numtlxpxls
                    
                    for col in range(0,tilecols):
                        
                        tile = onerow[:,tlidxlo:tlidxhi]
                        
                        print("tile size is {}".format(tile.shape))
                        
                        # Optional: plot tiles to check
                        plt.imshow(tile, cmap="gray")
                        plt.title("{}x{} tile".format(numtlxpxls,numtlypxls) + "\n" + "image {} ".format(img_num) + "tile {}".format(cut_tile_num))
                        plt.show()
                        
                        cut_tile_num += 1
                        
                        tiles_all_imgs.append(tile)
                        
                        tlidxlo += tlxoffset
                        tlidxhi += tlyoffset
                        
                    rowidxlo += tlyoffset
                    rowidxhi += tlxoffset
                    
                img_num += 1
                    
    # Convert to numpy array
    tiles_all_imgs = np.array(tiles_all_imgs, dtype=list)

    print("size of tiles all images: {}".format(tiles_all_imgs.shape))
    print("size of tiles all images[0] (first image's first tile): {}".format(tiles_all_imgs[0].shape) + "\n")
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
        if __name__ == "__data__":
            dataset_in = open(desired_dataset, "rb")
            X, y = pickle.load(dataset_in)
            dataset_in.close()

            print("\n" + "I. Dataset " + desired_dataset + " successfully loaded from local dir" + "\n")

        # I.e. if __name__ == "__preprocessing__", or we are in some other script
        else:
            print("\n" + "I. Desired dataset " + desired_dataset + " already present in local dir: would you like to overwrite it? (y/n)")
            ans = input()
            # For overwrite
            if ans == "y":
                # Create dataset per specifications
                X, y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
                dataset_out = open(desired_dataset, "wb")
                pickle.dump((X, y), dataset_out)
                dataset_out.close()

            else:
                print("Quitting dataset creation..." + "\n")
                exit()

    else:
        X, y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
        dataset_out = open(desired_dataset, "wb")
        pickle.dump((X, y), dataset_out)
        dataset_out.close()

    return X, y


####

### If you want to preprocess and pickle a dataset outside of a main.py model-training operation, or just overwrite an old one, run this script.

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


# Main will grab globals and prepare your data.
def main():

    print("Running data.py to generated desired dataset outside of main.py")
    X, y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
    return X, y


if __name__ == '__main__':

    X, y = main()
