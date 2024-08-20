from ast import literal_eval
from model import PredictiveCodingClassifier, StaticPCC, TiledStaticPCC, RecurrentPCC, TiledRecurrentPCC
from data import load_data
from os.path import join, exists
from os import listdir
from pickle import load

def load_params(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore lines that begin with '#'
            if not line.strip().startswith('#'):
                # Assuming each line in the file is in the format 'variable_name="value"'
                variable, value = line.strip().split('=', 1)
                # Use literal_eval to safely evaluate the value string
                evaluated_value = literal_eval(value)
                # Store the evaluated value in the params dictionary
                params[variable] = evaluated_value
    return params

def model_name_from_params(params):
    '''
    format 
    mod.TYPE_TILED_NUMLYRS_LR1SIZE...-LRNSIZE_NUMIMGS_ACTIVFUNC_PRIORS_CLASSIFMETHOD...
    ..._UPDATEMETHOD_CONFIGNAME_EP.pydb
    '''
    
    '''
    could optimize order later
    '''
    
    # LR1SIZE...-LRNSIZE
    layer_sizes = ''
    for l in range(params['num_layers']-1):
        size = params['hidden_lyr_sizes'][l]
        layer_sizes += str(size) + '-'
    layer_sizes += str(params['output_lyr_size'])
    
    # Get num imgs from desired training set
    num_imgs = params['dataset_train'].split('_')[1]
    
    # Assuming params['update_method'] is a dictionary with exactly one key-value pair
    update_method, um_int = next(iter(params['update_method'].items()))
    um_int_str = str(um_int)  # Convert the value to a string

    name = 'mod.' + params['model_type'] + '_' + ('tl' if params['tiled'] else 'ntl') + '_' \
                    + str(params['num_layers']) + layer_sizes + '_' + num_imgs + '_' + params['activ_func'] + '_' \
                    + params['priors'] + '_' + params['classif_method'] + '_' + \
                    + update_method + '-' + um_int_str + '_' + params['name'] + '_' + str(params['epoch_n']) + '.pydb'
                    
    return name

def load_model(model_path):
    '''
    Load a model from a filename using pickle.
    '''
    if not exists(model_path):
        raise ValueError(f"The model path {model_path} does not exist.")
    
    with open(model_path, 'rb') as file:
        model = load(file)
    return model

def load_checkpoint(model_name, params):
    '''
    load a checkpoint from the model name
    '''
    
    # loading will be the same up until : max epoch or chosen epoch step
    checkpoint_dir = join('models', 'checkpoints')
    model_name_no_epoch_no_pydb = model_name.rsplit('_', 1)[0]
    
    chk_dir_names = listdir(checkpoint_dir)
    matching_fns = [f for f in chk_dir_names if f.startswith(model_name_no_epoch_no_pydb)]
    if not matching_fns:
        raise ValueError('No matching checkpoint found')
    
    if params['load_checkpoint'] == -1:
        
        # case where we load the latest checkpoint
        max_epoch = -1
        chk_file_path = None
        file_name_valid = False
        
        for filename in matching_fns:
            try:
                # take off .pydb
                filename_no_pydb = filename.rsplit('.', 1)[0]
                # isolate epoch number
                epoch = int(filename_no_pydb.rsplit('_', 1)[1])
                # filename was right
                file_name_valid = True
                
                if epoch > max_epoch:
                    max_epoch = epoch
                    chk_file_path = join(checkpoint_dir, filename)
            except ValueError:
                continue
            
    elif params['load_checkpoint'] != -1:
        
        desired_epoch = params['load_checkpoint']
        chk_file_path = None
        file_name_valid = False
        
        for filename in matching_fns:
            try:
                # take off .pydb
                filename_no_pydb = filename.rsplit('.', 1)[0]
                # isolate epoch number
                epoch = int(filename_no_pydb.rsplit('_', 1)[1])
                # filename was right
                file_name_valid = True
                
                if epoch == desired_epoch:
                    chk_file_path = join(checkpoint_dir, filename)
            except ValueError:
                continue
            
    if not file_name_valid:
        raise ValueError('No valid checkpoint filenames found')
    
    checkpoint = load_model(chk_file_path)
    
    return checkpoint
            
def instantiate_model(params):
    
    # Base class
    model = PredictiveCodingClassifier()
    model.set_model_attributes(params)
    
    if model.model_type == 'static' and model.tiled == False:
        
    
    # Just the subclass
    if params['type'] == 'static' and params['tiled'] == False:
        model_init = StaticPCC
    elif params['type'] == 'static' and params['tiled'] == True:
        model_init = TiledStaticPCC
    elif params['type'] == 'recurrent' and params['tiled'] == False:
        model_init = RecurrentPCC
    elif params['type'] == 'recurrent' and params['tiled'] == True:
        model_init = TiledRecurrentPCC
    else:
        raise ValueError('Invalid model type')
    
    # Now attributes are params
    model = PredictiveCodingClassifier()
    model = set_model_attributes(model_init, params)
    model.validate_attributes()
    return model

def set_model_attributes(model, params):
    '''
    Set model attributes from the params dictionary.
    '''
    
    pass

def run_experiment(config_file_path):
    params = load_params(config_file_path)

    model_name = model_name_from_params(params)
    
    if params['train']:
        
        # Model
        if params['load_checkpoint'] is not None:
            model = load_checkpoint(model_name, params)
        else:
            model = instantiate_model(params['model_type'], params['tiled'])
    
        # Data
        X_train, Y_train = load_data(params['dataset_train'])
        
        # Train: will shuffle data automatically
        model.train(X_train, Y_train, save_checkpoint=params['save_checkpoint'], plot=params['plot_train'])
        
        '''
        later remove load_checkpoint from train- this is an outside function
        '''
        
    if params['evaluate']:
        
        mod_file_path = join('models', model_name)
        model = load_model(mod_file_path)
        
        # Evaluate will not
        X_eval, Y_eval = load_data(params['dataset_eval'])
        model.evaluate(X_eval, Y_eval, plot=params['plot_eval'])
        
    if params['predict']:
        
        mod_file_path = join('models', model_name)
        model = load_model(mod_file_path)
        
        # Predict will not
        X_pred = load_data(params['dataset_pred'])
        model.predict(X_pred, plot=params['plot_pred'])

    return None

def main(config_file_path=join('config', 'config_2024_07_22.txt')):
    '''
    change the second part of the path to the config file you want to use
    '''
    run_experiment(config_file_path)

if __name__ == '__main__':
    main()