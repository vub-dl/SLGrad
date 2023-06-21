# Dynamic Weighting Algorithms
This folder contains all code related to the dynamic weighting algorithms used to perform the experiments on the multi-task datasets. Currently, the code includes the following algorithms:

- **SLGrad:** The sample level task weighting algorithm proposed in our paper: https://arxiv.org/pdf/2306.04519.pdf. For using this algorithm, one should specify the size of the validation batch in the parameter dictionary and fill in AlgType.SLGrad.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary.  [Own implementation]

- **Static Weighting:** A standard baseline with static and uniformly assigned task weights. For usage, specify AlgType.Unif.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary. [Own implementation]

- **Random Weighting (RW):**  A simple weighting method where a model is trained with randomly assigned weights [[1]](#1).  For usage, specify AlgType.Random.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary. [Own_implementation]

- **Cosine Similarity (CosSim):** This algorithm only considers auxiliary task signals when the corresponding gradient aligns with the main task gradient direction [[2]](#2). For usage, specify AlgType.Gcosim.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary.  [Own_implementation]

- **OL-AUX:** An online learning algorithm that uses auxiliary task gradient directions, computed on previous batches to update auxiliary task weights. The gradient directions are computed on past batches and updated with gradient descent every few steps [[3]](#3). For usage, specify AlgType.Olaux.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary.  [Own_implementation]

- **PCGrad:** An algorithm that projects conflicting gradient directions of a task to the normal plane of any other task it negatively interferes with [[4]](#4). For usage, specify AlgType.PCGrad.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary. [Code adaption from LibMTL [[5]](#5)]

- **CAGrad:** Conflict-Averse Gradient descent that aims to simultaneously minimize the average loss and leverages the worst local improvement of individual tasks [[6]](#6). For usage, specify AlgType.CAgrad.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary. [Code adaption from LibMTL [[5]](#5)]

- **GradNorm:** Gradient normalization algorithm that automatically tunes gradient magnitudes to dynamically balance training in deep multi-task models [[7]](#7).  For usage, specify AlgType.Gnorm.value as a value for the 'Task_Weighting_strategy' in the parameter dictionary.[Own_implementation]



# References
<a id="1">[1]</a> 
Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W. Tsang (2021). 
Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task
Learning. 
arXiv: 2111.10603.

<a id="2">[2]</a> 
Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Mehrdad
Farajtabar, Razvan Pascanu, and Balaji Lakshminarayanan (2018). 
Adapting auxiliary losses using gradient similarity. 
arXiv: 1812.02224

<a id="3">[3]</a>
Xingyu Lin, Harjatin Baweja, George Kantor, and David Held (2019). 
Adaptive auxiliary task weighting for reinforcement learning. 
*Advances in Neural Information Processing Systems, 32*. 

<a id="4">[4]</a>
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol
Hausman, and Chelsea Finn (2020).
Gradient surgery for multi-task learning.
*Advances in Neural Information Processing Systems, 33*

<a id="5">[5]</a> 
Baijiong Lin and Yu Zhang (2022). 
LibMTL: A Python Library for Multi-Task Learning. 
arXiv preprint arXiv:2203.14338_.
Github: https://github.com/median-research-group/LibMTL/tree/0aaada50cd609b39c65553d4c2760c18b02d8e74/examples/nyu 

<a id="6">[6]</a> 
Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu (2021). 
Conflict-averse gradient descent for multi-task learning. 
*Advances in Neural Information Processing Systems, 34*

<a id="7">[7]</a> 
Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich (2018). 
Gradnorm: Gradient normalization for adaptive loss balancing
in deep multitask networks. 
*International Conference on Machine Learning*
