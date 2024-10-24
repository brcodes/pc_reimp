from ast import literal_eval
from model import PredictiveCodingClassifier, StaticPCC, RecurrentPCC
# from data import load_data
from os.path import join, exists
from os import listdir, makedirs
from pickle import load
import numpy as np
from datetime import datetime

def load_params(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace
            stripped_line = line.strip()
            # Ignore lines that begin with '#' or are empty
            if stripped_line and not stripped_line.startswith('#'):
                # Ensure the line contains an '=' character
                if '=' in stripped_line:
                    variable, value = stripped_line.split('=', 1)
                    # Use literal_eval to safely evaluate the value string
                    evaluated_value = literal_eval(value.strip())
                    # Store the evaluated value in the params dictionary
                    params[variable.strip()] = evaluated_value
                else:
                    raise ValueError(f"Invalid line in config file: {line}")
    return params

def model_name_from_params(params):
    '''
    format 
    mod.TYPE_TILED_NUMLYRS_LR1SIZE...-LRNSIZE_NUMIMGS_ACTIVFUNC_PRIORS_CLASSIFMETHOD...
    ..._UPDATEMETHOD_EXPNAME_EP.pydb
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
    num_inps = params['dataset_train'].split('_')[1]
    
    # Assuming params['update_method'] is a dictionary with exactly one key-value pair
    update_method, um_int = next(iter(params['update_method'].items()))
    um_int_str = str(um_int)  # Convert the value to a string
    architecture = params.get('architecture')

    name = 'mod.' + params['model_type'][:2] + '_' + ('tl' if params['tiled_input'] else 'ntl') + '_' \
                    + str(params['num_layers']) + '_' + layer_sizes + '_' + num_inps + '_' \
                    + ''.join([word[0] for word in architecture.split('_')]) + '_' \
                    + params['activ_func'][:3] + '_' \
                    + params['priors'] + '_' + params['classif_method'] + '_' \
                    + update_method + '-' + um_int_str + '_' + params['exp_name'] + '_' + str(params['epoch_n']) + '.pydb'
                    
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
        
        print('Loading latest checkpoint')
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
                    max_epoch = chk_epoch = epoch
                    chk_file_path = join(checkpoint_dir, filename)
            except ValueError:
                continue
            
    elif params['load_checkpoint'] != -1:
        
        print(f'Loading checkpoint at epoch {params["load_checkpoint"]}')
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
                    chk_epoch = desired_epoch
            except ValueError:
                continue
            
    if not file_name_valid:
        raise ValueError('No valid checkpoint filenames found')
    
    checkpoint = load_model(chk_file_path)
    
    print(f'Loaded checkpoint: {chk_file_path}')
    
    return checkpoint, chk_epoch
            
def instantiate_model(params):
    
    # Base class
    PCC = PredictiveCodingClassifier()
    PCC.set_model_attributes(params)
    PCC.config_from_attributes()
    
    if PCC.model_type == 'static':
        model = StaticPCC(PCC)
    elif PCC.model_type == 'recurrent':
        model = RecurrentPCC(PCC)
    else:
        raise ValueError('Invalid model type')
    
    return model

def create_data(dataset_name):
    '''
    Create data from scratch using desired parameters scraped from filename.
    '''
    
    # Get parameters from filenam
    # return x , y 
    pass

def load_data(dataset_name):
    '''
    Load data from a filename using pickle.
    Create it if it does not exist.
    '''
    data_path = join('data', dataset_name)
    if not exists(data_path):
        print(f"The data path {data_path} does not exist.")
        # Create
        X, Y = create_data(dataset_name)
    else:
        # Load
        with open(data_path, 'rb') as file:
            X, Y = load(file)
        
    return X, Y

def print_params(params):
    for key, value in params.items():
        print(f'{key}: {value}')
        
def initiate_log(params):
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    log_dir = 'log'
    makedirs(log_dir, exist_ok=True)
    log_file_name = f'exp_{timestamp}.txt'
    log_file_path = join(log_dir, log_file_name)
    with open(log_file_path, 'w') as log_file:
        # Init log
        log_file.write(f'Experiment log begin: {log_file_path}\n')
        for key, value in params.items():
            log_file.write(f'{key}: {value}\n')
    return log_file_name

def init_log_print_params(params):
    log_file_name = initiate_log(params)
    print_params(params)
    initiated_log = True
    return log_file_name, initiated_log

def run_experiment(config_file_path):
    params = load_params(config_file_path)

    model_name = model_name_from_params(params)
    
    initiated_log = False
    
    if params['train_with']:
        
        # Model
        if params['load_checkpoint'] is not None:
            model, epoch = load_checkpoint(model_name, params)
            load_name_ep_params = {'load_name': model_name, 'load_epoch': epoch, 'config_epoch_n': params['epoch_n']}
            model.set_model_attributes(load_name_ep_params)
            print(f'Desired final state: {model_name}\n')
        else:
            model = instantiate_model(params)
            model_name_ep_param = {'mod_name': model_name, 'config_epoch_n': None}
            model.set_model_attributes(model_name_ep_param)
            print(f'Instantiated model for desired final state: {model_name}\n')
    
        # Print and log params (base log for appending)
        log_file_name, initiated_log = init_log_print_params(params)
        log_file_name_param = {'exp_log_name': log_file_name}
        model.set_model_attributes(log_file_name_param)
        
        # Data
        X_train, Y_train = load_data(params['dataset_train'])
        print(f'Loaded data: {params["dataset_train"]}')
        
        # Train: will shuffle data automatically
        model.train(X_train, Y_train, save_checkpoint=params['save_checkpoint'], load_checkpoint=params['load_checkpoint'], plot=params['plot_train'])
        
        '''
        later remove load_checkpoint from train- this is an outside function
        '''
        
    if params['evaluate_with']:
        
        mod_file_path = join('models', model_name)
        model = load_model(mod_file_path)

        if not initiated_log:
            # Print and log params (base log for appending)
            log_file_name, initiated_log = init_log_print_params(params)
            log_file_name_param = {'exp_log_name': log_file_name}
            model.set_model_attributes(log_file_name_param)
        
        # Evaluate will not
        X_eval, Y_eval = load_data(params['dataset_eval'])
        model.evaluate(X_eval, Y_eval, plot=params['plot_eval'])
        
    if params['predict_with']:
        
        mod_file_path = join('models', model_name)
        model = load_model(mod_file_path)
        
        if not initiated_log:
            # Print and log params (base log for appending)
            log_file_name, initiated_log = init_log_print_params(params)
            log_file_name_param = {'exp_log_name': log_file_name}
            model.set_model_attributes(log_file_name_param)
        
        # Predict will not
        X_pred = load_data(params['dataset_pred'])
        model.predict(X_pred, plot=params['plot_pred'])

    return None

def main():
    '''
    change the second part of the path to the config file you want to use
    '''
    config_folder = 'config'
    if not exists(config_folder):
        raise ValueError(f"The config folder {config_folder} does not exist.")
    
    config_file_base = 'config_Li_rPCC'
    
    multi = False
    
    if multi:
        
        first = 'a'
        final = 'b'
        
        while first <= final:
            config_file_name = f'{config_file_base}{first}.txt'
            config_file_path = join(config_folder, config_file_name)
            
            if not exists(config_file_path):
                raise ValueError(f"The config file {config_file_path} does not exist.")
            
            print(f'Running experiment with config file: {config_file_path}')
        
            run_experiment(config_file_path)
            
            # Move to the next letter
            first = chr(ord(first) + 1)
            
    else:
        
        config_file_name = config_file_base + '.txt'
        config_file_path = join(config_folder, config_file_name)
        
        if not exists(config_file_path):
                raise ValueError(f"The config file {config_file_path} does not exist.")
            
        print(f'Running experiment with config file: {config_file_path}')
    
        run_experiment(config_file_path)

if __name__ == '__main__':
    main()