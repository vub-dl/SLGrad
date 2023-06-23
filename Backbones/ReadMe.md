# Backbones: overview

This folder contains the code related to the neural networks used as a backbone in the experiments performed on the different datasets. Currently, the folder provides code for conducting experiments with the following backbones:

- **MTNET:** standard neural network with a flexible number of shared and task-specific layers for multiple regression tasks. Used for experiments on Toy data.  

- **MTAN:** code adapted from the @LibMTL repository [[1]](#1). Used for experiments on NYUv2 [[2]](#2) multi-task dataset. 

- **LeNet**: LeNet-5 architecture used for experiments on the multi-task version of CIFAR-10 [[3]](#3). All layers are used as a shared encoder. Fully connected layers were added as task-specific functions with ReLU activation functions.

- **Multi-LeNet**: Slightly adapted LeNet architecture for experiments on the Multi-MNIST [[4]](#4) dataset. Code adapted from [[5]](#5). 


When choosing a dataset in the parameters in run_experiments.py, the correct backbone is automatically selected. To change the default backbone used for each dataset, modify the code in MTL_MODEL_OPT.py. 

# Extension

To add a new backbone to this repository, the following steps should be taken:

1. Add a new subclass to the "Backbone" superclass. Add new properties and attributes if necessary.
2. Make sure the layers are collected in the torch.nn.modules with names including "shared" and "task" for the shared and task-specific layers. This is needed for compatibility with the dynamic weighting algortihms.
3. MTL_OPT.py: connect the correct dataset to the new backbone in the **__init__** part
4. MTL_OPT.py: make sure the correct losses are selected for training this backbone
5. MTL_OPT.py > GetLosses: check if the predictions are passed to the loss functions in a correct way
6. run_experiments.py: enable the selection of this new backbone through the parameter configuration dictionary

If the addition of this backbone is combined with a new dataset, check the extension steps in the "Dataset" folder. 

# References

<a id="1">[1]</a> 
Baijiong Lin and Yu Zhang (2022). 
LibMTL: A Python Library for Multi-Task Learning. 
arXiv preprint arXiv:2203.14338_.
Github: https://github.com/median-research-group/LibMTL/tree/0aaada50cd609b39c65553d4c2760c18b02d8e74/examples/nyu 

<a id="2">[2]</a> 
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus (2012). 
Indoor Segmentation and Support Inference from RGBD Images. 
In *European Conference on Computer Vision*.

<a id="3">[3]</a> 
Alex Krizhevsky (2009). 
Learning multiple layers of features from tiny images. 
Technical report. 

<a id="4">[4]</a> 
Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton (2017).
Dynamic routing between capsules. 
*Advances in neural information processing systems*, **30**. 

<a id="5">[5]</a> 
Ozan Sener and Vladlen Koltun (2018).
Multi-Task Learning as Multi-Objective Optimization. 
*Advances in neural information processing systems*, **31**. 

