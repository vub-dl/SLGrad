# SLGRAD
This repository provides code for the experiments in **Sample-Level Weighting for Multi-Task Learning with Auxiliary Tasks** (https://arxiv.org/pdf/2306.04519).

<img width="412" alt="Capture" src="https://github.com/vub-dl/SLGrad/assets/108074099/2069dfcc-d7cc-4099-aa37-a8a48ef7d8c1">


## Setup

Before running experiments, make sure the required dependencies are installed:

```
pip install -r requirements.txt 
```

To run the experiments on your own device, make sure to unpack all files (folders as well). 

The results of the experiments will be automatically logged to Weights and Biases https://wandb.ai/site. Make sure to create an account before starting the experiments.   
## Usage

To run benchmark experiments with different dynamic weighting algorithms on different (semi) synthetic and real-world datasets, run the **run_experiments.py** file. 

Before running the code, specify the configuration in the dictionary provided in the  **run_experiments.py** file. The meaning of each parameter is discussed below. 

For example configurations, please check the **Examples** folders. 

### Parameters

- **Task_Weighting_strategy**: specifies the dynamic weighting method.
  
  - Supported values: AlgType.SLgrad.value, AlgType.Unif.value, AlgType.Olaux.value, AlgType.CAgrad.value, AlgType.Gnorm.value, AlgType.PCGrad.value

- **Dataset**: specifies the dataset the experiments are conducted on.

  - Supported values: DataName.Multi_MNIST.value, DataName.CIFAR10.value, DataName.Toy_reg.value, DataName.NYUv2.value
    
- **Number_of_Tasks**: specifies the number of tasks to train. Note that the "main task" automatically corresponds to task 0.

  - Supported values: [1, inf[ for DataName.Toy_reg.value     and  dataset specific for CIFAR10, Multi_MNIST and NYUv2.
 
- **input_dimension**: specifies the input dimension of the tensors.
  
  - Supported values: [1, inf[ in principle. See examples for specific experiments.

-  **output_dimension_task1** and **output_dimension_task2**: specify the output dimensions of task 1 and 2 respectively (for toy experiments only).

    - Supported values: [1, inf[ in principle. See examples for specific experiments.

-  **Batch_Size** and **val_Batch_Size**: specify the batch size of the training and validation set respectively. If task weighting strategy not equal to SLGrad, the     batch_size should correspond to the size of the validation set
    - Supported values: [1, inf[ in principle: to be optimized.
 
-  **Number_of_Shared_Layers** and **Dim_of_Shared_Layers**: specify the number and dimension of backbone layers shared by all tasks.

    - Supported values: [1, inf[ in principle: to be optimized. See examples and paper for specific experiments.

-  **Number_of_Task_Layers** and **Dim_Task_Layers**: specify the number and dimension of task specific backbone layers.

   -  Supported values: [1, inf[ in principle: to be optimized. See examples and paper for specific experiments.

- **Optimizer**: specifies the torch optimizer used to train the models.

   - Supported values: 'sgd' or 'adam' which will initialize torch.optim.SGD and torch.optim.Adam respectively

- **beta_1_backbone** and **beta_2_backbone**: specifies the beta 1 and beta 2 values for torch.optim.Adam if necessary.
  
    - Supported values: ]0, 0.99], to be optimized.

- **Learning_Weight**: specifies the learning rate used by the optimizer

    - Supported values: any appropriate learning rate, to be optimized. See examples for specific experiments

- **Onlymain**: specifies whether only the main task is noisy or all tasks. Only to be specified when "noise" > 0

    - Supported values: True, False

- **Noise**: specifies percentage of noise (only compatible with Toy and CIFAR10 experiments)

    - Supported values: [0,1]
 
- **random_seed**: specifies random state for reproducibility

    - Supported values: any positive integer

- **Regression**: specifies if generated toy data corresponds to regression tasks or classification tasks (to be specified only if Dataset is Toy_reg)

    - Supported values: True, False
      
- **UNI**: specifies if labels are flipped in uniform way or to background class (to be specified for CIFAR10 label flip experiments)

   - Supported values: True, False
  

## Overview

Updated soon.

## Citation

If you use this codebase or any part of it for a publication, please cite:

```
@misc{grégoire2023samplelevel,
      title={Sample-Level Weighting for Multi-Task Learning with Auxiliary Tasks}, 
      author={Emilie Grégoire and Hafeez Chaudhary and Sam Verboven},
      year={2023},
      eprint={2306.04519},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## References
Although most of the code has been implemented by us, we do rely on open-source repositories like [[1]] and [[2]].
We provide all references in the code-specific folders and thank all contributors for sharing their code. 

<a id="1">[1]</a> 
Baijiong Lin and Yu Zhang (2022). 
LibMTL: A Python Library for Multi-Task Learning. 
arXiv preprint arXiv:2203.14338_.
Github: https://github.com/median-research-group/LibMTL/tree/0aaada50cd609b39c65553d4c2760c18b02d8e74/examples/nyu 

<a id="2">[2]</a> 
Ozan Sener and Vladlen Koltun (2018).
Multi-Task Learning as Multi-Objective Optimization. 
*Advances in neural information processing systems*, **31**. 
Github: https://github.com/isl-org/MultiObjectiveOptimization.git

