import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def batch_data(data, bs=32):
    x, y = zip(*data) 
    batches =tf.data.Dataset.from_tensor_slices((list(x),list(y))).batch(bs)
    return batches 


def calculate_scale_factor(client_list, client):
    '''calculate scale factor = client_data size / all clients data size
    args:
        client_list: dict with clients names as keys, Batchdataset as items
        client: the client name of current client 
    return:
        numpy array : scalare'''
    
    #Get specific client number of samples    
    batch_size = len(list(client_list[client].as_numpy_iterator())[0][0])
    num_batches = tf.data.experimental.cardinality(client_list[client]).numpy()
    client_num_samples = batch_size*num_batches
    
    #count global samples used in this round (should be done in the server side in real life)
    server_num_samples = sum([tf.data.experimental.cardinality(client_list[client_name]).numpy()*len(list(client_list[client_name].as_numpy_iterator())[0][0]) for client_name in client_list.keys()])
    
    #Scale factor
    scale_factor = client_num_samples/server_num_samples
    
    return scale_factor


def scale_device_weights(s, w):
    #s: scalar , w: weights
    '''function for scaling a models weights'''
    weight_final = []
    num_layers = len(w)
    for i in range(num_layers):
        weight_final.append(s * w[i])
    return weight_final


def calculate_avg_weights(weights_list):
    
    avg_weights = list()
    
    #for each layer calculate sum
    for weights in zip(*weights_list):
        
        weights_sum = tf.reduce_sum(weights, axis =0)
        avg_weights.append(weights_sum)
    return avg_weights
    
    
    
    
def test_model(X_test, Y_test,  model, comm_round):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)

    #Convert Y_test into one hot encoded list
    lb = LabelBinarizer()
    Y_test = lb.fit_transform(Y_test)
    
    loss = loss_object(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss
    
    

def plot_clients_data_distribution(clients_data):
    
    
    f = plt.figure(figsize=(12, 7))
    f.suptitle('Label Counts for a Sample of Clients')
    for i in range(len(clients_data)):
        X_data, Y_data = zip(*clients_data[i])
        plt.subplot(2, 5, i+1)
        plt.title('Client {}'.format(i))
        plt.hist(Y_data,
                density=False,
                bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                rwidth= 0.5,
                histtype='barstacked')

        
       
    
    
    
    
    
    
    
    
    
    
    