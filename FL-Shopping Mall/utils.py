import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

import numpy as np

#Creat model
def MyModel(wind=12):
    model = Sequential()

    model.add(LSTM(200, activation='relu', return_sequences=True,input_shape=(wind, 1)))
    #model.add(Dropout(0.2))

    model.add(LSTM(100, activation='relu'))

    model.add(Dense(units = 1))
    return model


def calculate_scale_factor(client_list, client):
    '''calculate scale factor = client_data size / all clients data size
    args:
        client_list: dict with clients names as keys, data as items
        client: the client name of current client 
    return:
        numpy array : scalare'''
    
    #Get specific client number of samples    
    client_num_samples = client_list[client].shape[0]

    
    #count global samples used in this round (should be done in the server side in real life)
    server_num_samples = sum([client_list[client_name].shape[0] for client_name in client_list.keys()])
    
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


#evaluate model by calculating rmse

def evaluate_model(y_test, y_pred):
    
    #calculate score for whole prediction
    total_score = 0
    for row in range(y_test.shape[0]):
        for col in range(y_pred.shape[1]):
            total_score = total_score + (y_test[row, col] - y_pred[row, col])**2
    total_score = np.sqrt(total_score/(y_test.shape[0]*y_pred.shape[1]))
    
    return total_score
