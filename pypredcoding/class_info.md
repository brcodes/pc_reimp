To use our sPCC repo:


1. open config/config_YYYY_MM_DD.txt
2. edit parameters: 
    - parameters are parsed as literal python statements
	- e.g. input_size=(16,864) in config becomes self.input_size = (16,864) (a tuple) in the model.
    - lines starting with # are ignored.
    - right now, config only supports changes in:
        - exp_name : becomes the end of the model or diagnostic filenames
        - notes : a string to add more detail on an experiment
        - num_tiles : num tiles per input image
        - num_layers : can be >= 3, but not less than 3. have not played with > 3.
        - input_size : must be tuple of (numtilesperimage, sizeofflattenedtile) shape
        - hidden_lyr_sizes : [32,128] is Li model [r1,r2], and like hers, r1 actually becomes (num_tiles, hidden_lyr_sizes[0]) shape
        - output_lyr_size : 212 is Li model r3. If you're going to change this, and change c1/c2, make sure num_classes is what you want
        - classif_method : 'c1' or 'c2' or None. Haven't played with None, but it should be supported. Just reduces all c-cost contribs to 0.
        - priors : 'kurtotic' has her setup of r=0 U=uniform in [-0.5,0.5], 'gaussian' is standard normal, r,U.
        - update_method : {'rW_niters':30} i : r1-3 first, U1-3 second, for n = 30i. Any n supported.
                        : {'r_niters_W':100} i : r1-3 for n = 100i, then U1-3. Any n supported.
                        : {'r_eq_W':0.05} loop : changes in ea. r (1-3, individually) reduce to < 5% between iterations (L2-norm), then U1-3. Any % supported.
        - num_imgs : change this only if your dataset_train in data/dataset_train_name.pydb matches
        - num_classes : see output_lyr_size above.

    - Note: Only can use input dataset for training in format of: 

3. open run_experiment.py
4.a. set config_file_base to config_YYYY_MM_DD of choice, then run the script.
4.b. if multi = True, it will look for your filename base config_YYYY_MM_DD + letters from 'first' to 'final', and run them in sequence. Use generate_configs.py to generate config_YYYY_MM_DDa.txt, b.txt, c.txt etc. Instructions are there.




What's happening in run_experiment.py:


I. model parameters are assigned to PredictiveCodingClassifier base class
II. general stuff like update methods (eg rW30 vs r100W), prior functions, r and U initializations, .train(), evaluate(), etc
is created.
III. based on PCC.model_type == 'static' or 'recurrent', a StaticPCC or RecurrentPCC subclass is brought out, which inherits PCC attributes when PCC is fed to it as an (its only) argument. See below.

# Base class
    PCC = PredictiveCodingClassifier()
    PCC.set_model_attributes(params)
    PCC.config_from_attributes()
    
# Sub class
    if PCC.model_type == 'static':
        model = StaticPCC(PCC)
    elif PCC.model_type == 'recurrent':
        model = RecurrentPCC(PCC)
    else:
        raise ValueError('Invalid model type')
    
    # Now attributes are params
    model.validate_attributes()


IV. validate attributes is not yet written, but has been stubbed.


What's happening in model.py:

I. StaticPCC() has static-specific representation cost, classification cost, ri update equations, and rn update equations (etc)

NOTE: There are also differences in some of these equations between the n == 1, (number of layers == 1), 2, and 3+ cases.
This is because the 1st layer acts uniquely, as does the nth.

Thus, the equations you will want to look at are for n == 3 (number of layers == 3). 
DO NOT switch to less than 3 layers for your experiments.

II. These are called based on sPCC.attributes
III. Inherited general PCC.train() method (which is not called until necessary sPCC or rPCC-specific attributes are in place) is run, taking advantage of architecture-specific cost functions, update methods, etc.

NOTE:





