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
    base_config_date = '2024_09_16'
    base_config_name = 'config_' + base_config_date
    exp_name_base = 'notscaledbigr1_'
    
    base_config_path = 'config/' + base_config_name  + '.txt'
    kr_dict1 = kr={1:0.00001, 2:0.00001, 3:0.00001, 'o':0.00001}
    kr_dict2 = kr={1:0.0001, 2:0.0001, 3:0.0001, 'o':0.0001}
    # kU_dict1 = kU={1:0.001, 2:0.001, 3:0.001, 'o':0.001}
    # kU_dict2 = kU={1:0.01, 2:0.01, 3:0.01, 'o':0.01}
    rW_30 = {'rW_niters': 30}
    r_eq_W = {'r_eq_W': 0.05}
    r100W = {'r_niters_W': 100}
    
    param_dict = {
        'update_method': [ rW_30, rW_30, rW_30, r_eq_W, r100W],
        'kr': [kr_dict2, kr_dict1, kr_dict2, kr_dict2, kr_dict2],
        'classif_method': ['c1', 'c2', 'c2', 'c2', 'c2'],
        'exp_name': [exp_name_base + letter for letter in string.ascii_lowercase[:5]]
    }
    generate_configs(base_config_path, param_dict, base_config_name)
