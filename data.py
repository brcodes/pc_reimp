from keras.datasets import mnist
from keras.utils import np_utils
from numpy import ceil

def get_mnist_data(frac_samp=None,return_test=False):
    '''
    Returns MNIST training examples (and potentially test) data.  Images
    are flattened 28 x 28 images (784 pixels), and the target patterns are
    one-hot vectors.

    keras's mnist load returns 60,000 training samples and 10,000 test samples.
    Set frac_samp to a number in [0,1] to reduce the proportion of samples
    returned.
    '''
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    # flattening of images into vectors of pixels
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    # scaling
    # ?
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
        return (X_train[:n_train,:],y_train[:n_train,:]),(X_test[:n_test,:],y_test[:n_test,:])
    return X_train[:n_train,:],y_train[:n_train,:]
