import random
import numpy as np
from termcolor import colored

from tensorflow.keras.datasets import mnist

def generate_clients_data(num_clients, IsIID=True, batch_size=32):
    
    '''Takes in a full dataset and then assert each client a shard of data
    args:
        num_clients: integer number of clients
        Isiid: Bool to use IID or Non-IID data distrubution
    return:
        numpy array : clients data'''
        
    (x_train, y_train), (x_test, y_test)= mnist.load_data()

    x_train = x_train.astype('float32').reshape(-1,28*28)/255.0
    x_test = x_test.astype('float32').reshape(-1,28*28)/255.0

    training_data = list(zip(x_train, y_train))
    random.shuffle(training_data)

    if (IsIID == True):
        
        print(colored('---------- IID = True ----------', 'green'))  
        data_sz = len(training_data)
        size = data_sz//num_clients
        data_shards = [training_data[i:i+size] for i in np.arange(0, data_sz, size)]
        return data_shards, x_test, y_test
        
        
            
    else:
        ''' creates x non_IID clients'''

        
        
        print(colored('---------- IID = False ----------', 'green'))
        #create unique label list and shuffle
        
        unique_labels = np.unique(np.array(y_train))
        random.shuffle(unique_labels)
        
        class_shards = [None]*len(unique_labels)
        for item in unique_labels:
            #print(item)
            class_shards[item] = [(image, label) for (image, label) in zip(x_train, y_train) if label == item]
        
        
        return class_shards, x_test, y_test
    
    
        
    