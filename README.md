# FederatedLearning

An Implementation of the Federated Averaging Algorithm as described in the Paper - [Communication-Efficient Learning of Deep Networks from Decentralized Data by H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas](https://arxiv.org/abs/1602.05629).
Written in Python with support for TensorFlow and Keras. 

## Description

The main parameters to change in the main.py:
- <b>num_clients</b>: The number of devices participating in the process.
- <b>rounds</b>: The number of Communication rounds (number of global updates)
- <b>Epochs</b>: The number of training episodes within each device (number of local updates)
- <b>IsIID</b>: A Bool to choose whether to work on IID or non-IID data where:
<br> 1. The IID data looks like:

![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/figures/Figure%202020-11-13%20191459.png "Clients' data distribution")

<br> 2. The non-IID data looks like:

![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/figures/Figure%202020-11-13%20191511.png "")
