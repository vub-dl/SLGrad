# Backbones: overview

This folder contains the code related to the neural networks used as a backbone in the experiments performed on the different datasets. Currently, the following backbones:

- **MTNET:** standard neural network with a flexible number of shared and task-specific layers for multiple regression tasks. Used for experiments on Toy data.  

- **MTAN:** code adapted from the @LibMTL repository [[1]](#1). Used for experiments on NYUv2 [[2]](#2) multi-task dataset. 

- **LeNet**: LeNet-5 architecture used for experiments on the multi-task version of CIFAR-10 [[3]](#3). All layers are used as a shared encoder. Fully connected layers were added as task-specific functions with ReLU activation functions.

- **Multi-LeNet**: Slightly adapted LeNet architecture for experiments on the Multi-MNIST [[4]](#4) dataset. Code adapted from [[5]](#5). 


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

