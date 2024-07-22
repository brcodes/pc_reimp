# from keras.datasets import mnist, cifar10, fashion_mnist
# from keras.utils import np_utils
import numpy as np
import pickle
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

'''
li for ref
'''

def apply_DoG_filter(self, image, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        """
        Apply difference of gaussian (DoG) filter detect edge of the image.
        """
        g1 = cv2.GaussianBlur(image, ksize, sigma1)
        g2 = cv2.GaussianBlur(image, ksize, sigma2)
        return g1 - g2


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

        #NOTE: non-square sizing might be inverted, not tested.
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

    elif data_source == "trace212":
        # if non_mnist_images/trace212_{numxpxls}x{numypxls}/ is not present, quit
        desired_path = f"non_mnist_images/trace212_{numxpxls}x{numypxls}/"
        if not os.path.exists(desired_path):
            print(f" load raw images(): {desired_path} not found. quitting...")
            exit()
        print(f"{desired_path} found. loading...")
        # load all images in the directory
        
        image_names = []
        raw_imgs = []
        
        # List all files in the directory and save their names and images, checking for resolution parity with requested size
        for file in sorted(os.listdir(desired_path)):
            if file.endswith(".png"):
                # Extract name (everything before .png)
                name = file.split(".")[0]
                image_names.append(name)
                
                # Read and store the image
                img_path = os.path.join(desired_path, file)
                img = cv2.imread(img_path)
                
                
                # Check for parity with requested size
                # Note [1] is columns (x), [0] is rows [y]
                if img.shape[1] != numxpxls or img.shape[0] != numypxls:
                    print(f"load_raw_imgs(): trace212 image size (.pngs) does not match input size requested ({numxpxls}x{numypxls}). quitting...")
                    exit()
                    
                # Store the image
                raw_imgs.append(img)
                
            else:
                print(f"load_raw_imgs(): {file} is not a .png image. discluded from raw_imgs load...")
        
        print(f"load_raw_imgs(): {len(raw_imgs)} .png images loaded from {desired_path}")

    return np.array(raw_imgs)

def convert_to_gray(images):
    ### Only supports conversion from RGB (RB99, Rao99, CIFAR-10); MNIST, FMNIST already grayscale
    
    if len(images.shape) == 4:
        grayed_imgs = []
        for img in images:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            grayed_imgs.append(gray_img)

    # If already grayscale
    else:
        print("convert_to_gray(): input images detected to be already grayscale: input matrix shape is (numimgs,xpxls,ypxls" \
        " [vs RGB/BGR input matrix shape of (numimgs, xpxls, ypxls, 3)]. quitting...")
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

'''
Li for reference
'''

def create_gauss_mask(self, sigma=0.4, width=16, height=16):
        """ Create gaussian mask. """
        mu = 0.0
        x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
        d = np.sqrt(x**2+y**2)
        g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
        mask = g / np.max(g)
        return mask

        '''
        li for ref
        '''
        # Apply gaussian mask
        if self.use_mask:
            rf1_patch = rf1_patch * self.mask.reshape([-1])


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

'''
cut_tiles i beliece is deprecated
create_tiles is being tested for Li's 212 4x4 setup, and GENERALITY
cut_into_tiles has been tested for Li's 5 natimgs (15x15) setup and RB's 5 natimage (1x3) setup
'''

def create_tiles(images, numxpxls, numypxls, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):
    
    """
    Create tiles from the given images.

    This function is being tested for Li's 212 4x4 setup and for generality.
    It calculates the number of tiles in both x and y directions based on the provided
    dimensions and offsets, adjusts the offsets if necessary to ensure the tiles fit
    perfectly within the images, and then slices the images into tiles.

    Parameters:
    - numxpxls: The number of pixels in the x dimension of the images.
    - numypxls: The number of pixels in the y dimension of the images.
    - numtlxpxls: The width of the tiles in pixels.
    - numtlypxls: The height of the tiles in pixels.
    - tlxoffset: The offset between tiles in the x dimension.
    - tlyoffset: The offset between tiles in the y dimension.
    - images: A list of images to be tiled.

    Returns:
    An array of size (N images * n tiles, area of flattened tile) extracted from the images.
    Eg. 212 trace like inputs (212 images * 16 tiles per image = 3392 tiles, 36x24 = 864 pixels per tile)
    == 3392, 864
    """
    
    tiles = []
    for image in images:
        # Ensure the image is the correct size
        assert image.shape[0] == numypxls and image.shape[1] == numxpxls

        # Calculate the number of tiles in x and y directions
        num_xtiles = (numxpxls - numtlxpxls) // tlxoffset + 1
        num_ytiles = (numypxls - numtlypxls) // tlyoffset + 1

        # Adjust the offsets if the tiles do not fit perfectly within the image
        # Check if the total width of tiles is not equal to the image width
        if num_xtiles * tlxoffset + numtlxpxls != numxpxls:
            # Adjust tlxoffset to make the total width of tiles equal to the image width
            tlxoffset = (numxpxls - numtlxpxls) // (num_xtiles - 1)
            print(f"Adjusted tlxoffset to {tlxoffset} for horizontal alignment.")

        # Check if the total height of tiles is not equal to the image height
        if num_ytiles * tlyoffset + numtlypxls != numypxls:
            # Adjust tlyoffset to make the total height of tiles equal to the image height
            tlyoffset = (numypxls - numtlypxls) // (num_ytiles - 1)
            print(f"Adjusted tlyoffset to {tlyoffset} for vertical alignment.")
            
        # Create the tiles
        for i in range(num_xtiles):
            for j in range(num_ytiles):
                tile = image[j*tlyoffset:j*tlyoffset+numtlypxls, i*tlxoffset:i*tlxoffset+numtlxpxls]
                tiles.append(tile)

    # Convert to numpy array
    tiles_arr = np.array(tiles)

    print("size of tiles all images: {}".format(tiles_arr.shape))
    print("size of tiles all images[0] (first image's first tile before reshape): {}".format(tiles_arr[0].shape) + "\n")
    
    # reshape all tiles into 1D
    tiles_arr = tiles_arr.reshape(tiles_arr.shape[0], -1)
    
    print("size of tiles all images after 1d reshape: {}".format(tiles_arr.shape))
    print("size of tiles all images[0] (first image's first tile after reshape): {}".format(tiles_arr[0].shape) + "\n")
    
    return tiles_arr

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

        # If only horizontal offset (will default to RB99; 3 overlapping tiles centered on image center)
        if tlyoffset == 0 and tlxoffset != 0:
            tilecols = numtiles

            # Find image "center" using first image
            center_x = int(numxpxls / 2)
            center_y = int(numypxls / 2)

            # Tile 2 (center)
            tl2xidxlo = int(center_x - numtlxpxls / 2)
            tl2xidxhi = int(center_x + numtlxpxls / 2)
            tl2yidxlo = int(center_y - numtlypxls / 2)
            tl2yidxhi = int(center_y + numtlypxls / 2)
            # Tile 1 (center left)
            tl1xidxlo = tl2xidxlo - tlxoffset
            tl1xidxhi = tl2xidxhi - tlxoffset
            tl1yidxlo = tl2yidxlo
            tl1yidxhi = tl2yidxhi
            # Tile 3 (center right)
            tl3xidxlo = tl2xidxlo + tlxoffset
            tl3xidxhi = tl2xidxhi + tlxoffset
            tl3yidxlo = tl2yidxlo
            tl3yidxhi = tl2yidxhi

            # Initiate image counter for plot title
            img_num = 1

            for img in images:


                # New way: index by ys first (rows), then xs (cols)
                img_tl1 = img[tl1yidxlo:tl1yidxhi,tl1xidxlo:tl1xidxhi]
                img_tl2 = img[tl2yidxlo:tl2yidxhi,tl2xidxlo:tl2xidxhi]
                img_tl3 = img[tl3yidxlo:tl3yidxhi,tl3xidxlo:tl3xidxhi]


                """
                Optional: plot tiles to check cutting function fidelity
                """
                # plt.imshow(img_tl1, cmap="gray")
                # plt.title("{}x{} tile".format(numtlxpxls,numtlypxls) + "\n" + "image {} ".format(img_num) + "tile 1")
                # plt.show()
                #
                # plt.imshow(img_tl2, cmap="gray")
                # plt.title("{}x{} tile".format(numtlxpxls,numtlypxls) + "\n" + "image {} ".format(img_num) + "tile 2")
                # plt.show()
                #
                # plt.imshow(img_tl3, cmap="gray")
                # plt.title("{}x{} tile".format(numtlxpxls,numtlypxls) + "\n" + "image {} ".format(img_num) + "tile 3")
                # plt.show()

                # Array-ify and flatten the tiles for export to "X" (training input)
                img_tl1 = np.array(img_tl1).reshape(-1)
                img_tl2 = np.array(img_tl2).reshape(-1)
                img_tl3 = np.array(img_tl3).reshape(-1)

                tiles_all_imgs.append(img_tl1)
                tiles_all_imgs.append(img_tl2)
                tiles_all_imgs.append(img_tl3)

                img_num += 1

        # If only vertical offset
        elif tlyoffset != 0 and tlxoffset == 0:
            tilerows = numtiles
            print("cut_into_tiles(): vertical offset only, not yet written")
            exit()

        # If vertical and horizontal offset (RB97a: 4 tiles no overlap and Li 5 natimgs: 225 tiles, 8px x,y overlap)
        # Li 212 pseudospectrograms: 16 tiles, 36x24y, 32hoffset 20voffset. fully covers 132x84 image
        else:
            tilecols = int(np.sqrt(numtiles))
            tilerows = tilecols

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

                    # print("onerow size is {}".format(onerow.shape))

                    tlidxlo = 0
                    tlidxhi = numtlxpxls

                    for col in range(0,tilecols):

                        tile = onerow[:,tlidxlo:tlidxhi]

                        # print("tile size is {}".format(tile.shape))

                        """
                        Optional: plot tiles to check cutting function fidelity
                        """
                        plt.imshow(tile, cmap="cividis")
                        plt.title("{}x{} tile".format(numtlxpxls,numtlypxls) + "\n" + "image {} ".format(img_num) + "tile {}".format(cut_tile_num))
                        # add a colorbar
                        plt.colorbar()
                        plt.show()

                        # Array-ify and flatten the tile for export to "X" (training input)
                        tile = np.array(tile).reshape(-1)

                        tiles_all_imgs.append(tile)

                        cut_tile_num += 1

                        tlidxlo += tlxoffset
                        tlidxhi += tlyoffset

                    rowidxlo += tlyoffset
                    rowidxhi += tlxoffset

                img_num += 1

    # Convert to numpy array
    # Not sure why dtype = list was here. add back if necessary.
    # tiles_all_imgs = np.array(tiles_all_imgs, dtype=list)
    tiles_all_imgs = np.array(tiles_all_imgs)

    print("size of tiles all images: {}".format(tiles_all_imgs.shape))
    print("size of tiles all images[0] (first image's first tile): {}".format(tiles_all_imgs[0].shape) + "\n")
    return tiles_all_imgs

def preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset):

    # Labels (n, n), one hot diagonal from top left to bottom right
    y = np.eye(num_imgs)
    
    if data_source == "rb99":
        if num_imgs != 5:
            print("rb99 has 5 images; num_imgs must == 5. quitting...")
            exit()

        else:
            raw_imgs = load_raw_imgs(data_source, num_imgs, numxpxls, numypxls)

            if prepro == "lifull_lin":
                # Note that this RGB conversion is probably a function
                grayed_imgs = convert_to_gray(raw_imgs)
                # mask will take the whole image, or whole tile's dimensions
                gm_imgs = apply_gaussian_mask(grayed_imgs, numxpxls=numxpxls, numypxls=numypxls)
                dog_imgs = apply_DoG(gm_imgs)
                # Input
                X = dog_imgs

            elif prepro == "lifull_tanh":
                grayed_imgs = convert_to_gray(raw_imgs)
                # mask will take the whole image, or whole tile's dimensions
                gm_imgs = apply_gaussian_mask(grayed_imgs, numxpxls=numxpxls, numypxls=numypxls)
                dog_imgs = apply_DoG(gm_imgs)
                tanh_imgs = apply_tanh(dog_imgs)
                # Input
                X = tanh_imgs

            # RB99 orig prepro is not what Monica did
            # Theirs seems to be (p.86): 5 imgs first grayed, then whole images DoG'd, then tiles were individually Gaus.-masked

            elif prepro == "rb99full_lin":
                grayed_imgs = convert_to_gray(raw_imgs)
                # DoG images here are not final X, but will be passed through cut_into_tiles and then masked
                dog_imgs = apply_DoG(grayed_imgs)

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

            # If tiling desired, cut into tiles
            if tlornot == "tl":
                # All non RB99-original dset cases
                if prepro != "rb99full_lin" and prepro != "rb99full_tanh":
                    X = cut_into_tiles(X, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)

                # RB99 original dset: mask after cutting
                elif prepro == "rb99full_lin" or prepro == "rb99full_tanh":
                    cut_tiles = cut_into_tiles(dog_imgs, numxpxls, numypxls, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
                    # Need to change dimensions to tile dimensions
                    # might need to: Verify this last step in general!!! (2022.02.11)
                    X = apply_gaussian_mask(cut_tiles, numxpxls=numtlxpxls, numypxls=numtlypxls)
                    
    elif data_source == "trace212":
        if num_imgs != 212:
            print("trace212 has 212 images; num_imgs must == 212, quitting...")
            exit()
            
        raw_imgs = load_raw_imgs(data_source, num_imgs, numxpxls, numypxls)
        
        
        if prepro == "li_trace212":
            grayed_imgs = convert_to_gray(raw_imgs)
            # Make sure the mask is the same size as the tiles
            gm_imgs = apply_gaussian_mask(grayed_imgs, numxpxls=numxpxls, numypxls=numypxls)
            dog_imgs = apply_DoG(gm_imgs)
            
            if tlornot == "tl":
                X = create_tiles(dog_imgs, numxpxls, numypxls, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
            else:
                X = dog_imgs 
                
            # '''
            # example input
            # '''
            
            # cmap="cividis"

            # nrows = 1
            # ncols = 2
            # subplot_xy = (4.5, 3)
            # figsize = tuple([(ncols, nrows)[i]*subplot_xy[i] for i in range(2)])

            # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

            # ax1 = axes[0]
            # ax2 = axes[1]
            
            # print('printing example input')

            # sns.heatmap(grayed_imgs[0],
            #             cmap=cmap,
            #             cbar=True,
            #             cbar_kws={'shrink': 0.6},
            #             square=True,
            #             xticklabels=False,
            #             yticklabels=False,
            #             vmin=np.floor(np.min([x.min() for x in raw_imgs])),
            #             vmax=np.ceil(np.max([x.max() for x in raw_imgs])),
            #             ax=ax1);
            # ax1.set_title("Grayed(Original) Image");
            # # Adding text for min and max values of the first image
            # grayed_min, grayed_max = np.min(grayed_imgs[0]), np.max(grayed_imgs[0])
            # ax1.text(0.5, -0.1, f"Min: {grayed_min:.2f}", transform=ax1.transAxes, ha="center")
            # ax1.text(0.5, -0.15, f"Max: {grayed_max:.2f}", transform=ax1.transAxes, ha="center")


            # sns.heatmap(dog_imgs[0],
            #             cmap=cmap,
            #             cbar=True,
            #             cbar_kws={'shrink': 0.6},
            #             square=True,
            #             xticklabels=False,
            #             yticklabels=False,
            #             vmin=np.floor(np.min([x.min() for x in dog_imgs])),
            #             vmax=np.ceil(np.max([x.max() for x in dog_imgs])),
            #             ax=ax2);
            # ax2.set_title("Edge Detected(GMasked(Grayed(Original))) Image");
            # # Adding text for min and max values of the second image
            # dog_min, dog_max = np.min(dog_imgs[0]), np.max(dog_imgs[0])
            # ax2.text(0.5, -0.1, f"Min: {dog_min:.2f}", transform=ax2.transAxes, ha="center")
            # ax2.text(0.5, -0.15, f"Max: {dog_max:.2f}", transform=ax2.transAxes, ha="center")

                        
            # # idx 0
            # fig.suptitle("/brid/ ours")
            # # idx 30
            # # fig.suptitle("/b^s/")
            
            # fig.tight_layout(rect=[0,0,1,0.9]);
            # # Pdfs print weird with these, pngs fine. convert png later for paper if needed
            # # fig.savefig('example_input.pdf', dpi=800, bbox_inches='tight')
            # fig.savefig('brid ours example_input.png', dpi=600, bbox_inches='tight')

            
            # '''
            # all tiles of first image
            # '''
            # print('printing all tiles of first image')

            # # Loop for all rf2 patches
            # ## First image, S^t
            # rf1_patches = X[:16]
            # label = y[0]
            
            # # # Grab 31st image, b^s. This is what Monica titled her dissert ex image, but the image itself was S^t.
            # # rf1_patches = X[30*16:(30*16)+16]
            # # label = y[30]

            # for l, label in enumerate(y):
                
            #     if l == 46:
            #         rf1_patches = X[l*16:(l*16)+16]
            #         print(f'label {l} index', np.argmax(label))
            #         for p, patch in enumerate(rf1_patches):
                        
            #             print('nparray patch ')
            #             reshaped_patch = np.array(patch).reshape(24,36)
            #             print(reshaped_patch.shape)
            #             # Assuming 'patch' is a NumPy array representing the image data
            #             plt.figure()
            #             plt.imshow(reshaped_patch, cmap="cividis" )  # Use 'cmap' appropriate to your data
            #             plt.colorbar()
            #             # plt.title(f'S^t_rf1 patch {p}')
            #             # plt.savefig(f'S^t_rf1_patch_{p}.png')
            #             plt.title(f'brid_rf1_patch_{p}')
            #             plt.savefig(f'brid_rf1_patch_{p}.png')
            #             plt.show()
                    

            # exit()
        
        elif prepro is None:
            X = raw_imgs
    
    elif data_source == "mnist":
        print("preprocess(): mnist loading not yet written")
        exit()
        
    elif data_source == "fmnist":
        print("preprocess(): fmnist loading not yet written")
        exit()
        
    elif data_source == "cifar10":
        print("preprocess(): cifar10 loading not yet written")
        exit()
        
    elif data_source == "rao99":
        print("preprocess(): rao99 loading not yet written")
        exit()

    return X, y

def dataset_find_or_create(data_source="rb99", num_imgs=5, prepro="lifull",
    numxpxls=128, numypxls=128, tlornot="tl", numtiles=225,
    numtlxpxls=16, numtlypxls=16, tlxoffset=8, tlyoffset=8):

    ### Check for dataset in local directory: if present, load; if not, create, save for later
    # Default values are Li classification set values

    desired_dataset = "ds.{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pydb".format(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)

    print("I. Desired dataset is {}".format(desired_dataset) + "\n")

    if os.path.exists("./" + desired_dataset):

        print("Desired dataset " + desired_dataset + " already present in local dir: import it? (y/n)")

        ans = 'y'

        # For import
        if ans == "y":
            dataset_in = open(desired_dataset, "rb")
            X, Y = pickle.load(dataset_in)
            dataset_in.close()

            print("\n" + "Desired dataset " + desired_dataset + " successfully loaded from local dir" + "\n")

        # For overwrite or quit
        elif ans == "n":

            dataset_in = open(desired_dataset, "rb")
            X, Y = pickle.load(dataset_in)
            dataset_in.close()

            print("\n" + "Desired dataset " + desired_dataset + " import cancelled: create and overwrite instead? (y/n)")

            ans = input()

            # For overwrite
            if ans == "y":
            # Create dataset per specifications
                X, Y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
                print("Desired dataset " + desired_dataset + " successfully created" + "\n")
                dataset_out = open(desired_dataset, "wb")
                pickle.dump((X, Y), dataset_out)
                dataset_out.close()

                print("Desired dataset " + desired_dataset + " successfully pickled in local dir" + "\n")

            # For quit
            elif ans == "n":
                print("Need to either import a training dataset or create one: quitting main.py..." + "\n")
                exit()

    else:

        print("\n" + "I. Desired dataset " + desired_dataset + " is not present in local dir: create it? (y/n)")

        ans = input()

        if ans == "y":

            X, Y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
            print("\n" + "I. Desired dataset " + desired_dataset + " successfully created" + "\n")
            dataset_out = open(desired_dataset, "wb")
            pickle.dump((X, Y), dataset_out)
            dataset_out.close()

            print("\n" + "I. Desired dataset " + desired_dataset + " successfully pickled in local dir" + "\n")

        # For quit
        elif ans == "n":
            print("Need to either import a training dataset or create one: quitting main.py..." + "\n")
            exit()

    return X, Y, desired_dataset


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
    X, Y = preprocess(data_source, num_imgs, prepro, numxpxls, numypxls, tlornot, numtiles, numtlxpxls, numtlypxls, tlxoffset, tlyoffset)
    return X, Y


if __name__ == '__main__':

    X, Y = main()
