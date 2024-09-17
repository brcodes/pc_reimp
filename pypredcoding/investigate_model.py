import pickle
from os.path import join
from os import listdir
from matplotlib import pyplot as plt

# # Define the model path
# model_name = "mod.static_tl_3_32-128-212_212_linear_kurtotic_c1_r_niters_W-100_scaledbigr1_a_1.pydb"
# model_path = join('models', model_name)

# # Load the model
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)
    
# print(f'loaded model: {model}')

# # Print the accuracy
# print(f'self.accuracy {model.accuracy}')
# print(f'self.Jr {model.Jr}')
# print(f'self.Jc {model.Jc}')



# Directory containing the diagnostics
diags_dir = 'models'

# Iterate through all files in the models directory
for filename in listdir(diags_dir):
    # Check if the file starts with 'diag.'
    if filename.startswith('diag.'):
        diags_path = join(diags_dir, filename)
        
        # Load the model
        with open(diags_path, 'rb') as file:
            diags = pickle.load(file)
        
        print(f'loaded diag file: {filename}')
        
        epochs = range(len(diags['Jr']))

        # Calculate percent changes
        percent_change_Jr = ((diags['Jr'][-1] - diags['Jr'][0]) / diags['Jr'][0]) * 100
        percent_change_Jc = ((diags['Jc'][-1] - diags['Jc'][0]) / diags['Jc'][0]) * 100
        percent_change_accuracy = ((diags['accuracy'][-1] - diags['accuracy'][0]) / diags['accuracy'][0]) * 100

        fig, ax1 = plt.subplots()

        # Plot Jr
        ax1.plot(epochs, diags['Jr'], 'y-', label='Jr')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Jr', color='y')
        ax1.tick_params(axis='y', labelcolor='y')
        ax1.set_ylim(0, max(diags['Jr']))

        # Create a second y-axis for Jc
        ax2 = ax1.twinx()
        ax2.plot(epochs, diags['Jc'], 'b-', label='Jc')
        ax2.set_ylabel('Jc', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(0, max(diags['Jc']))

        # Create a third y-axis for accuracy
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(epochs, diags['accuracy'], 'r-', label='Accuracy')
        ax3.set_ylabel('Accuracy', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_ylim(0, max(diags['accuracy']))

        # Add legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')

        # Add percent change text
        plt.text(0.05, 0.95, f'Jr % Change: {percent_change_Jr:.2f}%', transform=ax1.transAxes, color='y')
        plt.text(0.05, 0.90, f'Jc % Change: {percent_change_Jc:.2f}%', transform=ax1.transAxes, color='b')
        plt.text(0.05, 0.85, f'Accuracy % Change: {percent_change_accuracy:.2f}%', transform=ax1.transAxes, color='r')

        plt.title('Training Diagnostics\n' + filename)
        plt.show()
        
        
        
        

