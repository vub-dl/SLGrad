# DATA: Overview

This folder contains the code related to datasets used for the multi-task experiments. 
Currently includes the following datasets/generators:

- **Toy Regression data generator**: subclass ToyRegDataset enables the creation of a synthetic multi-task dataset with or without additional noise. For usage, write DataName.Toy_reg.value as a value for "Dataset" in the parameter dictionary. 

- **NYUv2**: subclass NYU enables the selection of the original NYUv2 [[1]](#1). Code modified from [[2]](#2). For usage, write DataName.NYUv2.value as a value for  "Dataset" in the parameter dictionary. The download of the dataset is necessary before using the code. 

- **Multi-Task Cifar-10**: subclass CIFAR10 enables the transformation of the original CIFAR10  [[3]](#3) single-task dataset to its multi-task version. Generates 10 different binary tasks. The FlipLabels attributes enable uniform or background label flips when the noise parameter in the parameter dictionary is not equal to zero. For usage, write DataName.CIFAR10.value as a value for "Dataset" in the parameter dictionary. The dataset is imported from torchvision to avoid downloading the dataset. 

- **Multi-MNIST**: subclass Multi_MNIST enables the transformation from a subset of the original MNIST dataset [[4]](#4) to its multi-task version. Code adapted from [[5]](#5). For usage, write DataName.Multi_MNIST.value as a value for "Dataset" in the parameter dictionary. The dataset is imported from torchvision to avoid downloading the dataset. 


All datasets are transformed into Pytorch "Dataset" objects to enable compatibility with Pytorch's DataLoaders. For more information about Pytorch Datasets and DataLoaders, please check https://pytorch.org/tutorials/beginner/basics/data_tutorial.html. 


# References 

<a id="1">[1]</a> 
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus (2012). 
Indoor Segmentation and Support Inference from RGBD Images. 
In *European Conference on Computer Vision*.

<a id="2">[2]</a> 
Baijiong Lin and Yu Zhang (2022). 
LibMTL: A Python Library for Multi-Task Learning. 
arXiv preprint arXiv:2203.14338_.
Github: https://github.com/median-research-group/LibMTL/tree/0aaada50cd609b39c65553d4c2760c18b02d8e74/examples/nyu 

<a id="3">[3]</a> 
Alex Krizhevsky (2009). 
Learning multiple layers of features from tiny images. 
Technical report. 

<a id="4">[4]</a> 
Li Deng (2012). 
The mnist database of handwritten digit images for machine learning research’. 
*IEEE Signal Processing Magazine*, 29(6), 141–142.

<a id="5">[5]</a> 
Ozan Sener and Vladlen Koltun (2018).
Multi-Task Learning as Multi-Objective Optimization. 
*Advances in neural information processing systems*, **31**. 
