import os
import cv2
from parameters import ModelParameters, LR_params_to_dict, size_params_to_p_format
from model_draft import PredictiveCodingClassifier, model_find_and_or_create
from data import preprocess, dataset_find_or_create
import pickle
from sys import exit


""" Train a Predictive Coding Classifier according to the mathematical dictates
of Rao and Ballard 1999 """


def main():

    """
    main.py contents

    I. The Input Set

    1. Name input dataset parameters
    2. Check to see if requested dataset exists in local dir
        i. If it does, import it
        ii. If it does not, create it, save it

    II. The Model

    1. Name model parameters (not exhaustive: some model parameters will be called from section I.1)
    2. Check to see if requested model exists in local dir
        i. If it does, print that it does and either overwrite or terminate program
        ii. If it does not, initialize the model

    III. Training

    1. Train the model on the dataset
        NOTE:
        i. Model automatically pickles desired training checkpoints
        ii. Model generates a terminal readout for time taken and estimated time remaining
        iii. Training profile (cProfile) saved upon completion in a separate file
    """

    ####

    """
    I. The Input Set

    1. Name input dataset parameters
    2. Check to see if requested dataset exists in local dir
        i. If it does, import it
        ii. If it does not, create it, save it

    """

    # Dataset naming format is:
    # "source_numimgs_preprocessingscheme_numxpixls_numypixls_tilesornot_numtiles_numtilexpxls_numtileypxls_tilexoffset_tileyoffset.pydb"
    # E.g. Li's successful classification set of 5 imgs:
    # Rao and Ballard 1999, Li full suite of prepro, 128x128, 225 15x15 tiles, horizontal and vertical offset of 8
    # Would be: "rb99_5_lifull_128_128_tl_225_15_15_8_8.pydb"
    # RB99's would ~ be: "rb99_5_lifull_512_408_tl_3_16_16_5_0.pydb"

    ####

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

    ### Check for dataset in local directory: if present, load; if not, create, save for later

    X_train, y_train = dataset_find_or_create(data_source=data_source, num_imgs=num_imgs, prepro=prepro,
        numxpxls=numxpxls, numypxls=numypxls, tlornot=tlornot, numtiles=numtiles,
        numtlxpxls=numtlxpxls, numtlypxls=numtlypxls, tlxoffset=tlxoffset, tlyoffset=tlyoffset)

    ####

    """
    II. The Model

    1. Name model parameters (not exhaustive: some model parameters will be called from section I.1)
    2. Check to see if requested model exists in local dir
        i. If it does, print that it does and either overwrite or terminate program
        ii. If it does not, initialize the model
    """

    ### Set some model parameters for directory search

    ## Number of hidden layers
    # num_nonin_lyrs = 1
    # num_nonin_lyrs = 2
    num_nonin_lyrs = 3

    ## Layer (r) sizes
    lyr_sizes = (96, 128, 5)

    ## Num r[1] modules (= numtiles)
    num_r1_mods = numtiles

    ## Activation function
    act_fxn = "lin"
    # act_fxn = "tan"

    ## r, U priors
    # r_prior, U_prior = "gaus", "gaus"
    r_prior, U_prior = "kurt", "kurt"

    ## Classification paradigm
    # class_scheme = "nc"
    class_scheme = "c1"
    # class_scheme = "c2"

    ## Number of epochs to train
    # num_epochs = 0
    num_epochs = 500
    # num_epochs = 1000

    ### Set some more model parameters for model creation

    ## Learning rate scheme
    # lr_scheme = "constant"
    # lr_scheme = "poly"
    lr_scheme = "step"

    ## Initial learning rates
    ## r
    r_init = 0.005

    ## U
    U_init = 0.01

    ## o (only used during C2 classification)
    o_init = 0.00005


    ## Polynomial decay LR schedule
    ## r
    r_max_eps = num_epochs
    r_poly_power = 1

    ## U
    U_max_eps = num_epochs
    U_poly_power = 1

    ## o
    o_max_eps = num_epochs
    o_poly_power = 1


    ## Step decay LR schedule (d_f = 1 means LR is constant; d_e 40 epochs is Li's number)
    ## r
    r_drop_factor = 1
    r_drop_every = 40

    ## U (0.98522 = 1/1.015; Li's LR divisor for classifying linear PC model)
    # U_drop_factor = 1
    U_drop_factor = 0.98522
    U_drop_every = 40
 
    ## o
    o_drop_factor = 1
    o_drop_every = 40

    ### Automatically arrange LR parameters dict for p object, model creation

    k_r_sched, k_U_sched, k_o_sched = LR_params_to_dict(lr_scheme=lr_scheme, r_init=r_init, U_init=U_init, o_init=o_init,
    r_max_eps=r_max_eps, U_max_eps=U_max_eps, o_max_eps=o_max_eps,
    r_poly_power=r_poly_power, U_poly_power=U_poly_power, o_poly_power=o_poly_power,
    r_drop_factor=r_drop_factor, U_drop_factor=U_drop_factor, o_drop_factor=o_drop_factor,
    r_drop_every=r_drop_every, U_drop_every=U_drop_every, o_drop_every=o_drop_every)

    ### Automatically set the remainder of model parameters for parameters (p) object, model creation

    input_size, hidden_sizes, output_size = size_params_to_p_format(num_nonin_lyrs=num_nonin_lyrs,
    lyr_sizes=lyr_sizes, num_imgs=num_imgs, numxpxls=numxpxls, numypxls=numypxls)

    ## Parameters object creation
    p = ModelParameters(input_size = input_size, hidden_sizes = hidden_sizes, output_size = output_size,
        num_r1_mods = num_r1_mods, act_fxn = act_fxn, r_prior = r_prior, U_prior = U_prior,
        class_scheme = class_scheme,
        k_r_sched = k_r_sched,
        k_U_sched = k_r_sched,
        k_o_sched = k_o_sched)


    ### Helper fxn: search local dir for requested model, if it exists: initialize (for overwrite) or abort;
    ### if it doesn't exist, initialize

    def model_find_and_or_create(num_nonin_lyrs=3, lyr_sizes=(96,128,5), num_r1_mods=225, act_fxn="lin",
        r_prior="kurt", U_prior="kurt", class_scheme="c1", num_epochs=500):

        ### Directory search for named model

        # Initiate model name string
        desired_model = "mod.{}_".format(num_nonin_lyrs)

        if len(lyr_sizes) != num_nonin_lyrs:
            print("Number of non-input layers (num_nonin_lyrs) must == length of lyr_sizes tuple")
            exit()

        for lyr in range(0,num_nonin_lyrs):
            str_lyr = str(lyr_sizes[lyr])
            if lyr < num_nonin_lyrs - 1:
                desired_model += (str_lyr + "-")
            else:
                desired_model += (str_lyr + "_")

        ### Check for model in local directory: if present, quit (creation / training not needed); if not, create

        desired_model += "{}_{}_{}_{}_{}_{}.pydb".format(num_r1_mods, act_fxn, r_prior, U_prior, class_scheme, num_epochs)

        print("II. Desired model is {}".format(desired_model) + "\n")

        if os.path.exists("./" + desired_model):
            print("Desired model " + desired_model + " already present in local dir: would you like to overwrite it? (y/n)")
            ans = input()
            # For overwrite
            if ans == "y":
                # Initialize model

                mod = PredictiveCodingClassifier(p)

            else:
                print("Quitting main.py..." + "\n")
                exit()
        # For first save
        else:
            # Initialize model
            mod = PredictiveCodingClassifier(p)

        return mod

    ## Model object creation
    mod = model_find_and_or_create(num_nonin_lyrs=num_nonin_lyrs, lyr_sizes=lyr_sizes, num_r1_mods=num_r1_mods,
        act_fxn=act_fxn, r_prior=r_prior, U_prior=U_prior, class_scheme=class_scheme, num_epochs=num_epochs)

    ####

    """
    III. Training

    1. Train the model on the dataset
        NOTE:
        i. Model automatically pickles desired training checkpoints
        ii. Model generates a terminal readout for time taken and estimated time remaining
        iii. Training profile (cProfile) saved upon completion in a separate file
    """

    ### Train

    # train on training set
    mod.train(X_train, y_train)


if __name__ == '__main__':

    main()
