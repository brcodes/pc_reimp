import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import getctime, join, isfile
from datetime import datetime

# Directory containing the diagnostics
diags_dir = 'models'
filename_marker = ''

# # Which diagnostics to plot
# # Grab most recent in folder
which = 'most_recent'
# # Grab only those from today
# which = 'today'
# # Grab only those with a certain filename component
# which = 'filename_marker'
# filename_marker += 'softmax_k='
# Grab all
# which = 'all'

# Get the list of files in the directory
files = listdir(diags_dir)

# Filter files based on the 'which' parameter
if which == 'most_recent':
    print(f'grabbing most recent file from {diags_dir}')
    files = [max(files, key=lambda x: getctime(join(diags_dir, x)))]
elif which == 'today':
    print(f'grabbing files from today in {diags_dir}')
    today = datetime.today().strftime('%Y-%m-%d')
    files = [f for f in files if datetime.fromtimestamp(getctime(join(diags_dir, f))).strftime('%Y-%m-%d') == today]
elif which == 'filename_marker':
    print(f'grabbing files with {filename_marker} in {diags_dir}')
    files = [f for f in files if filename_marker in f]
elif which == 'all':
    print('grabbing all files in {diags_dir}')
    pass  # No filtering needed

print(files)

# Iterate through the filtered files
for filename in files:
    # Check if the file starts with 'diag.'
    if filename.startswith('diag.'):
        diags_path = join(diags_dir, filename)
        
        # Load the model
        with open(diags_path, 'rb') as file:
            diags = pickle.load(file)
        
        print(f'loaded diag file: {filename}')
        
        epochs = range(len(diags['Jr']))
        
        # Turn each accuracy into an actual percent
        diags['accuracy'] = [acc * 100 for acc in diags['accuracy']]

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
        ax1.set_ylim(0, max(diags['Jr']) * 1.05)

        # Create a second y-axis for Jc
        ax2 = ax1.twinx()
        ax2.plot(epochs, diags['Jc'], 'b-', label='Jc')
        ax2.set_ylabel('Jc', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(0, max(diags['Jc']) * 1.05)

        # Create a third y-axis for accuracy
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(epochs, diags['accuracy'], 'r-', label='Accuracy', linewidth=0.5)
        ax3.set_ylabel('Accuracy (%)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_ylim(min(diags['accuracy']) * 1.05, max(diags['accuracy']) * 1.05)

        # Add legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='center left', bbox_to_anchor=(-.3, .9))

        # Add percent change text
        plt.text(0.1, 0.95, f'Jr % Change: {percent_change_Jr:.2f}%', transform=ax1.transAxes, color='y')
        plt.text(0.1, 0.90, f'Jc % Change: {percent_change_Jc:.2f}%', transform=ax1.transAxes, color='b')
        plt.text(0.1, 0.85, f'Accuracy % Change: {percent_change_accuracy:.2f}%', transform=ax1.transAxes, color='r')
        
        plt.title('Training Diagnostics\n' + filename)
        
        results_folder = 'results'
        results_path = join(results_folder, filename)
        fig.savefig(results_path + '.png', dpi=300, bbox_inches='tight')
        
        plt.show()