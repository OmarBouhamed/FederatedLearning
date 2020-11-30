# FederatedLearning

A basic implementation of the Federated Averaging Algorithm, written in Python with support for TensorFlow and Keras (Experiments are produced on ENERNOC 2012 Commercial Energy Consumption Data). 

## Description
Data, presenting the annual energy consumption of 9 Shppoing centers, was utilzed: 8 shopping centers hostory was used for the training process and the 9_th one was used as a test subject to evalute the global model. 

### Example of the data used:
![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/FL-Mnist/figures/IID.png)

## Results:
A Global model's RMSE eq ~0.05 for the non-participant client (9th shopping center) for 10 rounds of FL training and 5 epochs of local training.

![alt text](https://github.com/OmarBouhamed/FederatedLearning/blob/main/FL-Mnist/figures/IID.png)

## Reference

[Communication-Efficient Learning of Deep Networks from Decentralized Data by H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas](https://arxiv.org/abs/1602.05629).
