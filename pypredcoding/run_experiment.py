from ast import literal_eval
from model import StaticPCC, TiledStaticPCC, RecurrentPCC, TiledRecurrentPCC
from data import load_data
from os.path import join

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

def run_experiment(config_file_path):
    params = load_params(config_file_path)

    model = instantiate_model(params['model_type'], params['tiled'])

    X_train, Y_train = load_data(params['dataset_train'])
    # Train will shuffle data automatically
    model.train(X_train, Y_train, plot=True)
    # Evaluate will not
    X_eval, Y_eval = load_data(params['dataset_eval'])
    model.evaluate(X_eval, Y_eval, plot=params['eval_plot'])
    # Predict will not
    X_pred = load_data(params['dataset_pred'])
    model.predict(X_pred, plot=params['pred_plot'])

    return None

def main(config_file_path=join('config', 'config_2024_07_22.txt')):
    '''
    change the second part of the path to the config file you want to use
    '''
    run_experiment(config_file_path)

if __name__ == '__main__':
    main()