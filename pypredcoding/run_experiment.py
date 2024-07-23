from ast import literal_eval
from model import StaticPCC, TiledStaticPCC, RecurrentPCC, TiledRecurrentPCC
from data import load_data
from os.path import join
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
    ..._UPDATEMETHOD_CONFIGNAME_EP(_CHKifcheckpoint).pydb
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
                    + update_method + '-' + um_int_str + '_' + params['name'] + str(params['epoch_n']) + '_'
                    
    if params['load_checkpoint'] is not None:
        name += str(params['load_checkpoint'])
        
    return name

def load_model(model_path):
    '''
    Load a model from a filename using pickle.
    '''
    with open(model_path, 'rb') as file:
        model = load(file)
    return model
            
def instantiate_model(type, tiled):
    if type == 'static' and tiled == False:
        return StaticPCC
    elif type == 'static' and tiled == True:
        return TiledStaticPCC
    elif type == 'recurrent' and tiled == False:
        return RecurrentPCC
    elif type == 'recurrent' and tiled == True:
        return TiledRecurrentPCC
    else:
        raise ValueError('Invalid model type')
    
def set_model_attributes(model, params):
    '''
    Set model attributes from the params dictionary.
    '''
    pass

def run_experiment(config_file_path):
    params = load_params(config_file_path)

    model_name = model_name_from_params(params)
    
    if params['load_checkpoint'] is not None:
        if params['load_checkpoint'] == -1:
            '''
            case where we load the latest checkpoint
            '''
            chk_file_path=join('models', model_name)
            model = load_model(chk_file_path)
            
    '''
    format 
    mod.TYPE_TILED_NUMLYRS_LR1SIZE...-LRNSIZE_NUMIMGS_ACTIVFUNC_PRIORS_CLASSIFMETHOD...
    ..._UPDATEMETHOD_CONFIGNAME_EP(_CHKifcheckpoint).pydb
    '''
    
    # if isinstance(params['load_checkpoint'], int):
    #     if params['load_checkpoint'] == -1:
    #         chk_file_path=join('models', model_name)
    #         model = load_model(chk_file_path)
    #     elif params['load_checkpoint'] != -1:
    #         chk_file_path=join('models', model_name)
    #         model = load_model(chk_file_path)
    #     else:
    #         raise ValueError('Invalid load_checkpoint value')
    # elif isinstance(params['load_checkpoint'], None):
    #     model = instantiate_model(params['model_type'], params['tiled'])

    if params['train']:
        X_train, Y_train = load_data(params['dataset_train'])
        # Train will shuffle data automatically
        model.train(X_train, Y_train, save_checkpoint=params['save_checkpoint'], plot=params['plot_train'])
        
        '''
        later remove load_checkpoint from train- this is an outside function
        '''
    
    if params['evaluate']:
        # Evaluate will not
        X_eval, Y_eval = load_data(params['dataset_eval'])
        model.evaluate(X_eval, Y_eval, plot=params['plot_eval'])
        
    if params['predict']:
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