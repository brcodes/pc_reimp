import pickle
from os.path import join

# Define the model path
model_name = "mod.static_tl_3_32-128-212_212_linear_kurtotic_c2_r_niters_W-100_Experiment_2024_09_03_50.pydb"
model_path = join('models', model_name)

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
print(f'loaded model: {model}')

# Print the accuracy
print(f'self.accuracy {model.accuracy}')