# What's happening in run_experiment.py

I. model parameters are assigned to PredictiveCodingClassifier base class
II. general stuff like update methods (eg rW30 vs r100W), prior functions, r and U initializations, .train(), evaluate(), etc
is created.
III. based on PCC.model_type == 'static' or 'recurrent', a StaticPCC or RecurrentPCC subclass is brought out, which inherits PCC attributes when PCC is fed to it as an (its only) argument. See below.

## Base class

    PCC = PredictiveCodingClassifier()
    PCC.set_model_attributes(params)
    PCC.config_from_attributes()
    
## Sub class

    if PCC.model_type == 'static':
        model = StaticPCC(PCC)
    elif PCC.model_type == 'recurrent':
        model = RecurrentPCC(PCC)
    else:
        raise ValueError('Invalid model type')

What's happening in model.py:

I. StaticPCC() has static-specific representation cost, classification cost, ri update equations, and rn update equations (etc). These are loaded from cost_functions.py (StaticCostFunction class)

II. These are called based on sPCC.attributes

III. Inherited general PCC.train() method (which is not called until necessary sPCC or rPCC-specific attributes are in place) is run, taking advantage of architecture-specific cost functions, update methods, etc., as well as generalities.
