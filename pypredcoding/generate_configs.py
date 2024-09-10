import os
import ast

def read_config(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_config(lines, new_params, suffix):
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
    new_file_path = f"config/config_2024_09_10{suffix}.txt"
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines)

def generate_configs(base_config_path, param_dict):
    lines = read_config(base_config_path)
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    num_configs = len(values[0])
    
    for i in range(num_configs):
        new_params = {keys[j]: values[j][i] for j in range(len(keys))}
        suffix = chr(97 + i)  # 'a', 'b', 'c', etc.
        write_config(lines, new_params, suffix)

if __name__ == "__main__":
    base_config_path = 'config/config_2024_09_10.txt'
    kr_dict1 = kr={1:0.00001, 2:0.00001, 3:0.00001, 'o':0.00001}
    kr_dict2 = kr={1:0.0001, 2:0.0001, 3:0.0001, 'o':0.0001}
    rW_30 = {'rW_niters': 30}
    r_eq_W = {'r_eq_W': 0.05}
    r100W = {'r_niters_W': 100}
    
    param_dict = {
        'update_method': [rW_30, r_eq_W, r100W, rW_30, r_eq_W, r100W, rW_30, r_eq_W, r100W, rW_30, r_eq_W, r100W],
        'kr': [kr_dict1, kr_dict1, kr_dict1, kr_dict2, kr_dict2, kr_dict2, kr_dict1, kr_dict1, kr_dict1, kr_dict2, kr_dict2, kr_dict2], 
        'classif_method': ['c1', 'c1', 'c1', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2', 'c2', 'c2', 'c2']
    }
    generate_configs(base_config_path, param_dict)
