from keras.datasets import mnist
from keras.utils import np_utils
import cv2
import numpy as np


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
        return (X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:]).astype('float64'),(X_test[:n_test,:,:].astype('float64'),y_test[:n_test,:].astype('float64'))
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

    Standardizes a sample image (1D vector) by subtracting it's mean pixel value and
    then dividing by the standard deviation of the pixel values.

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
