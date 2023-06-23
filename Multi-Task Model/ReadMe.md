# Multi-Task Model

This folder contains the multi-task optimizer and fit function:

- MTL_MODEL_OPT.py: multi-task optimizer. Based on the parameters provided in the dictionary in run_experiments.py, this function connects the convenient dataset with the corresponding backbone, losses, and metrics. Furthermore, it collects the weights and the losses provided by the convenient dynamic weighting algorithm. 

- Experiment_Fit.py: custom fit function for the experiments. Calls the multi-task optimizer for each batch during the number of epochs specified by the parameter dictionary in run_experiments.py. Furthermore, all the necessary metrics are logged for wandb in this function. 

These files are automatically called when starting an experiment from run_experiments.py. To add datasets, baselines, or dynamic weighting algorithms, these files need to be adapted accordingly. 


## MTL_MODEL_OPT

The multi-task optimizer is the central object of the repository: it connects the dataset with the correct backbone and optimizes the backbone model in a way defined by the specific dynamic weighting algorithm selected in the configuration. 

The Global_MTL class consists of the following attributes:

- **__init__**: initialize the parameters of the multi-task optimizer based on the "configuration dictionary" passed by run_experiments. The hyperparameters and backbone (depending on the dataset), weighting method, and corresponding initial task weights are initialized here. Next, the losses and metrics are initialized: every experiment/type of task needs another loss function.
- 
- **setInvalid** is a function used to check if infinite values are occurring in one of the predicted tensors. If infinities are detected, they are replaced by finite values. This function is called by the **predict** function. 
  
- **predict**: a function that calls the forward method of the backbone for each task if task=-1 or only one task if task /= -1. 

- **process_preds**: only needed for experiments with MTAN backbone. Brings predictions in a convenient size to be compared with ground truth labels.

- **GetLoss**: computes the task and total losses. Uses the loss-type initialized at the start of the experiments by calling self.loss and self.lossElmt. Based on the parameters, the total loss is a weighted sum, a dot product of task weights with their corresponding task loss, or a weighted sum of sample-level task losses. The parameter values for "weighting", "elmt" and "task" are set in the **Evaluate** function, which contains different values for each dynamic weighting algorithm and train/val/test subset. When implementing a new dynamic weighting strategy, make sure the correct arguments are called such that the total loss is computed in the correct way. 

- **GetMetric**: idem as GetLoss but for the metric functions initialized at the beginning; If there is no difference between the loss and the metric, make sure that the **Evaluate function¨** only calls the loss to avoid double computations.

- **Evaluate**: a function that calls both the predict and getloss/getmetric function for a given batch of features and ground truth labels. Depending on the dynamic weighting algorithm and the subset (train/val/test), the parameters of the **predict**, **getloss** and **getmetric** functions will be different. Furthermore, these parameter values will determine whether the performed operations have to be saved or not (torch.no_grad).

- **train**: the general training loop, can be called by dynamic weighting algorithms. Some algorithms do not have a custom training loop (example, unif and random): they simply use the **train** function.
