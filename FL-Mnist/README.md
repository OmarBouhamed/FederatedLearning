# FederatedLearning

A basic implementation of the Federated Averaging Algorithm, written in Python with support for TensorFlow and Keras (Experiments are produced on MNIST)

- <b>IsIID</b>: A Bool to choose whether to work on IID or non-IID data where:
<br> 1. The IID data : Each client is randomly assigned a uniform distribution over 10 classes:

![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/FL-Mnist/figures/IID.png "Clients' data distribution")

<br> 2. The non-IID data : Each client receives data partitionfrom only a single class:

![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/FL-Mnist/figures/non_IID.png "")

## Reference

[Communication-Efficient Learning of Deep Networks from Decentralized Data by H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas](https://arxiv.org/abs/1602.05629).
