# To use our (currently only) sPCC repo

1. open config/config_YYYY_MM_DD.txt
2. edit parameters:
    - parameters are parsed as literal python statements
    - e.g. input_size=(16,864) in config becomes self.input_size = (16,864) (a tuple) in the model.
    - config will tell you what can't be changed (unsupported)
3. open run_experiment.py
4. set config_file_base to config_YYYY_MM_DD of choice, then run the script.
   - if multi = True, it will look for your filename base config_YYYY_MM_DD + letters from 'first' to 'final' (usually 'a' to something), and run them in sequence. Use generate_configs.py to generate config_YYYY_MM_DDa.txt, b.txt, c.txt etc. Instructions are there.

# The system is hard coded to save and plot a diagnostic file every 10 epochs right now

It will only save a model itself if it reaches the end of your regime (epoch == epoch_n).
"Diagnostics" is simply a dictionary containing keys 'Jr' 'Jc' and 'accuracy' whose values are lists with the epoch-averaged rep cost, class cost, acc. These are what is grabbed by the plotting code.
    - These lists include 'epoch 0' which is before any training.
    - 'Evaluations' are simply running the unshuffled 212 TRACE inputs through the model, with r but no weight updates, and calculating costs, accuracy.
    - 'Training' is shuffling those same inputs every epoch, running through them, etc. (our training and evaluation sets are identical, as of now.)
Find all model files in models/
    - mod.texttexttext for model files
    - diag.mod.textexttext for diagnostic files
Find assoc. diagnostic plots in results/
    - diag.mod.textexttext.PNG for images
  
Whenever an experiment is run, an experiment log is added in:
    - models/log/exp_YYMMDD_HHMM.txt (current datetime, down to minutes)
    - File is appended as the model experiment progresses, using PCC class method print_and_log()
      - Everything printed using print_and_log() (i) prints to console, and (ii) saves to exp_YYMMDD_HHMM.txt
      - This was easier and more helpful to me than other logging systems, ultimately.