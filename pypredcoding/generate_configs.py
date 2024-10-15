import os
import ast
import string

def read_config(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_config(lines, new_params, base_config_name, suffix):
    new_lines = []
    for line in lines:
        if line.strip().startswith('#') or '=' not in line:
            new_lines.append(line)
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key in new_params:
            new_value = new_params[key]
            if isinstance(new_value, str):
                new_lines.append(f"{key}='{new_value}'\n")
            else:
                new_lines.append(f"{key}={new_value}\n")
        else:
            new_lines.append(line)
    new_file_path = f"config/{base_config_name}{suffix}.txt"
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines)

def generate_configs(base_config_path, param_dict, base_config_name):
    lines = read_config(base_config_path)
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    num_configs = len(values[0])
    
    for i in range(num_configs):
        new_params = {keys[j]: values[j][i] for j in range(len(keys))}
        suffix = chr(97 + i)  # 'a', 'b', 'c', etc.
        write_config(lines, new_params, base_config_name, suffix)

if __name__ == "__main__":


    '''
    user sets: vvv
    '''
    # Base config you want to search over
    base_config_date = '2024_09_20'

    # Number of experiments (new configs from a to z to generate, based on base config)
    num_experiments = 2
    
    # Variables to set for those experiments, must be in list form. then will follow the literal eval sequence.
    classif_method = ['c1', 'c2']

    # Do not put in list form. this takes the general filename appendage for your search, and will auto.
    #-matically put a, b etc at the end of it, to differentiate runs. This becomes the model.exp_name parameter.
    exp_name_base = ''
    
    '''
    user sets: ^^^
    '''

    base_config_name = 'config_' + base_config_date
    base_config_path = 'config/' + base_config_name  + '.txt'


    '''
    user sets: vvv
    
    Add: 'variable_name' : variable_named_list_from_above (what you are searching over)
    Do not change: 'exp_name' code
    '''

    param_dict = {
        'classif_method': classif_method,
        'exp_name': [exp_name_base + letter for letter in string.ascii_lowercase[:num_experiments]]
    }
    
    generate_configs(base_config_path, param_dict, base_config_name)
