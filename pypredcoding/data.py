from keras.datasets import mnist
from keras.utils import np_utils
from numpy import ceil,prod


def get_mnist_data(frac_samp=None,return_test=False):
    '''
    Returns MNIST training examples (and potentially test) data.  Images are
    returned in an array of n_samples x 28 x 28, and target patterns are
    n_samples x num_classes one hot vectors.

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
        n_train = int(ceil(frac_samp*X_train.shape[0]))
        n_test = int(ceil(frac_samp*X_test.shape[0]))
    else:
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
    # send it all back
    if return_test:
        return (X_train[:n_train,:,:],y_train[:n_train,:]),(X_test[:n_test,:,:],y_test[:n_test,:])
    return X_train[:n_train,:,:],y_train[:n_train,:]


def flatten_images(image_array):
    '''
    Accepts an array of N x dim_x x dim_y images (N images each of dim_x x dim_y size)
    and returns a 2D array of flattened images

    This function will fail on a single image unless you give it an empty first
    axis (np.newaxis,:,:).
    '''
    N = image_array.shape[0]
    dim_x = image_array.shape[1]
    dim_y = image_array.shape[1]
    return image_array.reshape(N,dim_x*dim_y)


def inflate_vectors(vector_array,shape_2d=None):
    '''
    Accepts an array of N x s flattened images (vectors) and returns an array of
    N x shape_2d[0] x shape_2d[1] images.  If shape_2d is none, images are
    assumed to be square.

    This function will fail on a single input vector unless you give it an empty
    first axis (np.newaxis,:)
    '''
    N = vector_array.shape[0]
    if shape_2d == None:
        sq = int(sqrt(vector_array.shape[1]))
        shape_2d[0] = sq
        shape_2d[1] = sq
    return vector_array.reshape(N,shape_2d[0],shape_2d[1])
