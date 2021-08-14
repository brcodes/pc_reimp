from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils
import cv2
import numpy as np


def get_keras_data(dataset='mnist',frac_samp=None,return_test=False):
    '''
    Returns MNIST, Fashion MNIST, or CIFAR-10 training examples (and potentially test) data.
    Images (X_train and X_test) are returned in an array of n_samples x 28 x 28 (CIFAR-10: n x 32 x 32),
    and target patterns (y_train and y_test) are n_samples x num_classes-length one hot vectors.

    keras's mnist and fashion mnist load returns 60,000 training samples and 10,000 test samples.
    Set frac_samp to a number in [0,1] to reduce the proportion of samples
    returned. CIFAR-10 load returns 50,000 training samples and 10,000 test samples. frac_samp applies
    the same way here.
    '''
    if dataset == 'mnist':
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
            return (X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64')),(X_test[:n_test,:,:].astype('float64'),y_test[:n_test,:].astype('float64')), return_test
        return X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64'), return_test

    elif dataset == 'fashion_mnist':
        # read from Keras
        (X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
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
            return (X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64')),(X_test[:n_test,:,:].astype('float64'),y_test[:n_test,:].astype('float64')), return_test
        return X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64'), return_test

    elif dataset == 'cifar10':
        # read from Keras
        (X_train,y_train),(X_test,y_test) = cifar10.load_data()
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
            return (X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64')),(X_test[:n_test,:,:].astype('float64'),y_test[:n_test,:].astype('float64')), return_test
        return X_train[:n_train,:,:].astype('float64'),y_train[:n_train,:].astype('float64'), return_test

    else:
        print("get_keras_data() arg dataset must == 'mnist', 'fashion_mnist', or 'cifar10'")


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

def set_model_pkl_name(pc_obj, img_set_attrs=(), used_in_which_script='training',name_components_so_far=None, extra_tag):
    """ Sets output pickle name of a PC model object based on its class attributes,
    the image set trained, evaluated or predicted against, and an optional extra tag.
    Returns a 17 item tuple (the name components). used_in_which_script tells the function whether
    to associate img_set_attrs with a training, evaluation, or prediction image set. name_components_so_far
    allow image set name variables to "travel" with the model (e.g. from training to evaluation)
    and still be included in its name. """
    name_components = []

    #1.Tiling
    if pc_obj.is_tiled == True:
        tile_offset = str(pc_obj.p.tile_offset)
        tiled = "TL" + tile_offset
        name_components.append(tiled)
    elif pc_obj.is_tiled == False:
        tiled = "ntl"
        name_components.append(tiled)
    else:
        print("PC model object is_tiled must be either True or False")

    #2.Hidden layer sizes
    hidden_layers = "["
    #All but last layer
    for layer in pc_obj.p.hidden_sizes[:-1]:
        hidden_layer = str(layer)
        hidden_layers += hidden_layer + ","
    #Last layer
    hidden_layers += pc_obj.p.hidden_sizes[-1] + "]"
    name_components.append(hidden_layers)

    #3.Activation function
    if pc_obj.p.unit_act == 'tanh':
        act_fxn = 'tan'
    elif pc_obj.p.unit_act == 'linear':
        act_fxn = 'lin'
    else:
        print("PC model object p.unit_act must be either 'tanh' or 'linear'")
    name_components.append(act_fxn)

    #4.Priors
    priors = 'r'
    #r prior
    if pc_obj.p.r_prior == 'gaussian':
        priors += 'g'
    elif pc_obj.p.r_prior == 'kurtotic':
        priors += 'k'
    else:
        print("PC model object p.r_prior must be either 'gaussian' or 'kurtotic'")
    #U prior
    priors += 'U'
    if pc_obj.p.U_prior == 'gaussian':
        priors += 'g'
    elif pc_obj.p.U_prior == 'kurtotic':
        priors += 'k'
    else:
        print("PC model object p.U_prior must be either 'gaussian' or 'kurtotic'")
    name_components.append(priors)

    #5.Classification
    #can be 'NC', 'C1', or 'C2'
    class_type = pc_obj.p.classification
    name_components.append(class_type)

    #6.Trained or untrained
    #training_time = 0 unless the model has been trained, in which case it becomes
    #a tuple of 3 datetime objects.
    if type(pc_obj.training_time) == int:
        trained = 'nt'
    elif type(pc_obj.training_time) == tuple:
        trained = 'T'
    else:
        print("T or nt: PC model object training_time must be either 0 (int) or a tuple of datetime objects \
        in the format (tstart,tend,telapsed)")
    name_components.append(trained)

    #7.Batch size
    batch_size = 'b' + str(pc_obj.p.batch_size)
    name_components.append(batch_size)

    #8.Number of epochs
    num_epochs = str(pc_obj.p.num_epochs) + 'e'
    name_components.append(num_epochs)

    #9.Training image set, if any
    if trained == 'T':
        if used_in_which_script == 'training':
            tset = ''
            #Image source
            if img_set_attrs[0] == 'mnist':
                tset += 'M'
            elif img_set_attrs[0] == 'fashion_mnist':
                tset += 'FM'
            elif img_set_attrs[0] == 'cifar10':
                tset += 'C'
            else:
                print("tset: img_set_attrs[0] must be 'mnist', 'fashion_mnist', or 'cifar10'")
            #Preprocessing
            if img_set_attrs[1] == 'tanh':
                tset += 't'
            elif img_set_attrs[1] == 'linear':
                test += 'l'
            else:
                print("tset: img_set_attrs[1] must be either 'tanh' or 'linear'")
            #Size
            if img_set_attrs[2] == '28x28':
                tset += '28'
            elif img_set_attrs[2] == '24x24':
                tset += '24'
            else:
                print("tset: img_set_attrs[2] must be either '28x28' (for non-tiled model) or '24x24' (for tiled model)")
            #Number of images
            total_imgs = img_set_attrs[3]
            evenly_dist = img_set_attrs[4]
            num_imgs = ''
            if evenly_dist == True:
                num_imgs += str(total_imgs/10) + 'x' + '10'
            elif evenly_dist == False:
                num_imgs += str(total_imgs)
            else:
                print("tset: img_set_attrs[4] must be either True or False")
            tset += 'num_imgs'
            #From Train or Test set
            #attrs[5] should be 'tr' or 'ts'
            train_or_test = img_set_attrs[5]
            tset += train_or_test
            name_components.append(tset)
        elif used_in_which_script == 'evaluation' or used_in_which_script == 'prediction' or used_in_which_script == 'plotting':
            tset = name_components_so_far[8]
            name_components.append(tset)
        else:
            print("(training image set portion error) this function must be called in either the 'training' (main.py), 'evaluation', or 'prediction' script")
    elif trained == 'nt':
        tset = '-'
        name_components.append(tset)
    else:
        print("Set T-set name: the name component variable named 'trained' must be either 'T' or 'nt'")

    #10.Learning rate (called k, or LR) r
    k_r_list = list(self.p.k_r_sched.items())
    #Constant LR, Poly(nomial) decay, or Step decay
    k_r_type = k_r_list[0][0]
    #Initialize string
    k_r = 'kr'
    if k_r_type == 'constant':
        k_r += 'c'
        c_init = str(k_r_list[0][1][0])
        k_r += c_init
    elif k_r_type == 'poly':
        k_r += 'p'
        p_init = str(k_r_list[0][1][0])
        k_r += p_init
        max_ep = str(k_r_list[0][1][1])
        k_r += 'm' + max_ep
        pow = str(k_r_list[0][1][2])
        k_r += 'p' + pow
    elif k_r_type == 'step':
        k_r += 's'
        s_init = str(k_r_list[0][1][0])
        k_r += s_init
        drop_factor = str(k_r_list[0][1][1])
        k_r += 'd' + drop_factor
        drop_every = str(k_r_list[0][1][2])
        k_r += 'e' + drop_every
    else:
        print("k_r_type must be 'constant', 'poly', or 'step'")

    #11.Learning rate U
    k_U_list = list(self.p.k_U_sched.items())
    #Constant LR, Poly(nomial) decay, or Step decay
    k_U_type = k_U_list[0][0]
    #Initialize string
    k_U = 'kU'
    if k_U_type == 'constant':
        k_U += 'c'
        c_init = str(k_U_list[0][1][0])
        k_U += c_init
    elif k_U_type == 'poly':
        k_U += 'p'
        p_init = str(k_U_list[0][1][0])
        k_U += p_init
        max_ep = str(k_U_list[0][1][1])
        k_U += 'm' + max_ep
        pow = str(k_U_list[0][1][2])
        k_U += 'p' + pow
    elif k_U_type == 'step':
        k_U += 's'
        s_init = str(k_U_list[0][1][0])
        k_U += s_init
        drop_factor = str(k_U_list[0][1][1])
        k_U += 'd' + drop_factor
        drop_every = str(k_U_list[0][1][2])
        k_U += 'e' + drop_every
    else:
        print("k_U_type must be 'constant', 'poly', or 'step'")

    #12.Learning rate o (for classification type C1/C2)
    #will only add this to the filename if class_type is C1 or C2
    if class_type == 'C1' or 'C2':
        k_o_list = list(self.p.k_o_sched.items())
        #Constant LR, Poly(nomial) decay, or Step decay
        k_o_type = k_o_list[0][0]
        #Initialize string
        k_o = 'ko'
        if k_o_type == 'constant':
            k_o += 'c'
            c_init = str(k_o_list[0][1][0])
            k_o += c_init
        elif k_o_type == 'poly':
            k_o += 'p'
            p_init = str(k_o_list[0][1][0])
            k_o += p_init
            max_ep = str(k_o_list[0][1][1])
            k_o += 'm' + max_ep
            pow = str(k_o_list[0][1][2])
            k_o += 'p' + pow
        elif k_o_type == 'step':
            k_o += 's'
            s_init = str(k_o_list[0][1][0])
            k_o += s_init
            drop_factor = str(k_o_list[0][1][1])
            k_o += 'd' + drop_factor
            drop_every = str(k_o_list[0][1][2])
            k_o += 'e' + drop_every
        else:
            print("k_o_type must be 'constant', 'poly', or 'step'")
    elif class_type == 'NC':
        k_o = 'ko-'
    else:
        print("class_type for k_o assignment must be 'NC','C1', or 'C2'")

    #13.Evaluated or not
    #evaluation_time = 0 unless the model has been evaluated, in which case it becomes
    #a tuple of 3 datetime objects.

    if type(pc_obj.evaluation_time) == int:
        evaluated = 'ne'
    elif type(pc_obj.evaluation_time) == tuple:
        evaluated = 'E'
    else:
        print("E or ne: PC model object evaluation_time must be either 0 (int) or a tuple of datetime objects \
        in the format (tstart,tend,telapsed)")

    #14.Evaluation image set
    if evaluated == 'E':
        if used_in_which_script == "evaluation":
            eset = ''
            #Image source
            if img_set_attrs[0] == 'mnist':
                eset += 'M'
            elif img_set_attrs[0] == 'fashion_mnist':
                eset += 'FM'
            elif img_set_attrs[0] == 'cifar10':
                eset += 'C'
            else:
                print("eset: img_set_attrs[0] must be 'mnist', 'fashion_mnist', or 'cifar10'")
            #Preprocessing
            if img_set_attrs[1] == 'tanh':
                eset += 't'
            elif img_set_attrs[1] == 'linear':
                eset += 'l'
            else:
                print("eset: img_set_attrs[1] must be either 'tanh' or 'linear'")
            #Size
            if img_set_attrs[2] == '28x28':
                eset += '28'
            elif img_set_attrs[2] == '24x24':
                eset += '24'
            else:
                print("eset: img_set_attrs[2] must be either '28x28' (for non-tiled model) or '24x24' (for tiled model)")
            #Number of images
            total_imgs = img_set_attrs[3]
            evenly_dist = img_set_attrs[4]
            num_imgs = ''
            if evenly_dist == True:
                num_imgs += str(total_imgs/10) + 'x' + '10'
            elif evenly_dist == False:
                num_imgs += str(total_imgs)
            else:
                print("eset: img_set_attrs[4] must be either True or False")
            eset += 'num_imgs'
            #From Train or Test set
            #attrs[5] should be 'tr' or 'ts'
            train_or_test = img_set_attrs[5]
            eset += train_or_test
            name_components.append(eset)
        elif: used_in_which_script == 'prediction' or used_in_which_script == 'plotting' or used_in_which_script == 'training':
            eset = name_components_so_far[13]
            name_components.append(eset)
        else:
            print("(evaluation image set portion error) this function must be called in either the 'training' (main.py), 'evaluation', or 'prediction' script")
    elif evaluated == 'ne':
        eset = '-'
        name_components.append(eset)
    else:
        print("The name component variable named 'evaluated', as called during Evaluation image set name setup, must be either 'E' or 'ne'")

    #15.Used for Prediction, or not
    #prediction_time = 0 unless the model has been used for prediction, in which case it becomes
    #a tuple of 3 datetime objects.
    if type(pc_obj.prediction_time) == int:
        predicted_with = 'np'
    elif type(pc_obj.prediction_time) == tuple:
        predicted_with = 'P'
    else:
        print("P or np: PC model object prediction_time must be either 0 (int) or a tuple of datetime objects \
        in the format (tstart,tend,telapsed)")

    #16.Prediction image set
    if predicted_with == 'P':
        if used_in_which_script == "prediction":
            pset = ''
            #Image source
            if img_set_attrs[0] == 'mnist':
                pset += 'M'
            elif img_set_attrs[0] == 'fashion_mnist':
                pset += 'FM'
            elif img_set_attrs[0] == 'cifar10':
                pset += 'C'
            else:
                print("pset: img_set_attrs[0] must be 'mnist', 'fashion_mnist', or 'cifar10'")
            #Preprocessing
            if img_set_attrs[1] == 'tanh':
                pset += 't'
            elif img_set_attrs[1] == 'linear':
                pset += 'l'
            else:
                print("pset: img_set_attrs[1] must be either 'tanh' or 'linear'")
            #Size
            if img_set_attrs[2] == '28x28':
                pset += '28'
            elif img_set_attrs[2] == '24x24':
                pset += '24'
            else:
                print("pset: img_set_attrs[2] must be either '28x28' (for non-tiled model) or '24x24' (for tiled model)")
            #Number of images
            total_imgs = img_set_attrs[3]
            evenly_dist = img_set_attrs[4]
            num_imgs = ''
            if evenly_dist == True:
                num_imgs += str(total_imgs/10) + 'x' + '10'
            elif evenly_dist == False:
                num_imgs += str(total_imgs)
            else:
                print("pset: img_set_attrs[4] must be either True or False")
            pset += 'num_imgs'
            #From Train or Test set
            #attrs[5] should be 'tr' or 'ts'
            train_or_test = img_set_attrs[5]
            pset += train_or_test
            name_components.append(eset)
        elif: used_in_which_script == 'evaluation' or used_in_which_script == 'training' or used_in_which_script = 'plotting':
            pset = name_components_so_far[15]
            name_components.append(pset)
        else:
            print("(prediction image set portion error) this function must be called in either the 'training' (main.py), 'evaluation', or 'prediction' script")
    elif predicted_with == 'np':
        pset = '-'
        name_components.append(pset)
    else:
        print("The name component variable named 'predicted_with', as called during Prediction image set name setup, must be either 'P' or 'np'")

    #17.Extra tag
    extra_tag = extra_tag
    name_components.append(extra_tag)

    #Make immutable
    name_components = tuple(name_components)

    return name_components

def import_pickled_pc_model(tiled=False, toffset=6, hlsizes=[32,32],act_fxn='tanh',rprior='g',Uprior='g',class_type='NC',trained=True,batch_size=1,\
                num_epochs=40,tset='Mt28_100x10tr',kr=('c',0.05,None,None),kU=('c',0.05,None,None),ko=('c',0.05,None,None),evaluated=False,eset='Mt28_100x10tr',\
                predicted_with=False,pset='Mt28_100x10tr',extra_tag='-'):

    name_components_from_args = []

    #Tiling
    if tiled == True:
        tiling = 'TL' + str(toffset)
    else:
        tiling = 'ntl'
    name_components_from_args.append(tiling)

    #HLs
    hidden_layers = str(hlsizes)
    name_components_from_args.append(hidden_layers)

    #Activation
    name_components_from_args.append(act_fxn)

    #Priors
    priors = 'r'+rprior+'U'+Uprior
    name_components_from_args.append(priors)

    #Classification type
    name_components_from_args.append(class_type)

    #Trained or not
    if trained == True:
        train = 'T'
    else:
        train = 'nt'
    name_components_from_args.append(train)

    #Batch size
    batch = 'b'+str(batch_size)
    name_components_from_args.append(batch)

    #Num epochs
    epochs = str(num_epochs) + 'e'
    name_components_from_args.append(epochs)

    #Training set
    if trained == True:
        t_set = tset
    else:
        t_set = '-'
    name_components_from_args.append(t_set)

    #Learning rate r
    if kr[0] == 'c':
        lr_r = kr[0] + str(kr[1])
    elif kr[0] == 'p':
        lr_r = kr[0] + str(kr[1]) + 'm' + str(kr[2]) + 'p' + str(kr[3])
    else:
        lr_r = kr[0] + str(kr[1]) + 'd' + str(kr[2]) + 'e' + str(kr[3])
    name_components_from_args.append(lr_r)

    #Learning rate U
    if kU[0] == 'c':
        lr_U = kU[0] + str(kU[1])
    elif kU[0] == 'p':
        lr_U = kU[0] + str(kU[1]) + 'm' + str(kU[2]) + 'p' + str(kU[3])
    else:
        lr_U = kU[0] + str(kU[1]) + 'd' + str(kU[2]) + 'e' + str(kU[3])
    name_components_from_args.append(lr_U)

    #Learning rate o
    if ko[0] == 'c':
        lr_o = ko[0] + str(ko[1])
    elif ko[0] == 'p':
        lr_o = kr[0] + str(ko[1]) + 'm' + str(ko[2]) + 'p' + str(ko[3])
    else:
        lr_o = ko[0] + str(ko[1]) + 'd' + str(ko[2]) + 'e' + str(ko[3])
    name_components_from_args.append(lr_o)

    #Evaluated or not
    if evaluated == True:
        eval = 'E'
    else:
        eval = 'ne'
    name_components_from_args.append(eval)

    #Evaluation image set
    if evaluated == True:
        e_set = eset
    else:
        e_set = '-'
    name_components_from_args.append(e_set)

    #Predicted with or not
    if predicted_with == True:
        pred = 'P'
    else:
        pred = 'np'
    name_components_from_args.append(pred)

    #Prediction image set
    if predicted_with == True:
        p_set = pset
    else:
        p_set = '-'
    name_components_from_args.append(p_set)

    #Extra tag
    name_components_from_args.append(extra_tag)

    model_in = open('pc.pydb','rb')
    model = pickle.load(model_in)
    model_in.close()

    name_components_from_args.append(extra_tag)

    name = ''
    for component in name_components_from_args:
        name += component + '.'

    model_in = open('pc.{}pydb'.format(name),'rb')
    model = pickle.load(model_in)
    model_in.close()

    return model, name_components_from_args


def import_pickled_img_set(source='mnist',act_fxn='tanh',size='28x28',num_imgs=1000,evenly_dist=True,train_or_test='tr'):

    if evenly_dist == True:
        num_classes = 10
        num_instances = num_imgs / num_classes

        img_set_in = open('{}_{}_{}x{}_{}.pydb'.format(source,act_fxn,num_instances,num_classes,size),'rb')
        image_set = pickle.load(img_set_in)
        img_set_in.close()

        img_set_attrs = (source, act_fxn, size, num_imgs, evenly_dist, train_or_test)

        return image_set, img_set_attrs

    elif evenly_dist == False:

        img_set_in = open('{}_{}_{}_{}.pydb'.format(source,act_fxn,num_imgs,size),'rb')
        image_set = pickle.load(img_set_in)
        img_set_in.close()

        img_set_attrs = (source, act_fxn, size, num_imgs, evenly_dist, train_or_test)

        return image_set, img_set_attrs

    return image_set, img_set_attrs
