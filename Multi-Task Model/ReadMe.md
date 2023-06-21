# Multi-Task Model

This folder contains the multi-task optimizer and fit function:

- MTL_MODEL_OPT.py: multi-task optimizer. Based on the parameters provided in the dictionary in run_experiments.py, this function connects the convenient dataset with the corresponding backbone, losses, and metrics. Furthermore, it collects the weights and the losses provided by the convenient dynamic weighting algorithm. 

- Experiment_Fit.py: custom fit function for the experiments. Calls the multi-task optimizer for each batch during the number of epochs specified by the parameter dictionary in run_experiments.py. Furthermore, all the necessary metrics are logged for wandb in this function. 

These files are automatically called when starting an experiment from run_experiments.py. To add datasets, baselines, or dynamic weighting algorithms, these files need to be adapted accordingly. 
