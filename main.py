import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from termcolor import colored

from generate_clients_data import generate_clients_data
from utils import *
from model import *


num_clients = 10
IsIID = False
rounds = 10
Epochs = 5

clients_data, x_test, y_test = generate_clients_data(num_clients, IsIID)

#create testing sets
test_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

#plot data destrib
plot_clients_data_distribution(clients_data)

client_names = ['{}_{}'.format('client', i+1) for i in range(num_clients)]

client_list = dict(zip(client_names, clients_data))
    
for client in client_list:
    client_list[client] = batch_data(client_list[client])
print(colored('----------Data batched----------', 'green'))    
    
    
#Create global model for the server
global_model = MyModel()

#Specify model compile settings:
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True) #use softmax
optimizer=keras.optimizers.Adam(lr=0.001)
metrics=['accuracy']

#Federated Learning process

#for each round of communication
for rnd in range(rounds):
    
    #to initialize the local models
    global_weights = global_model.get_weights()
    
    #shuffle local devices: for randomness
    random.shuffle(client_names)
    
    #list of scaled devices_model weights after training
    list_Scaled_weights = list()
    
    #for each client participating in the round
    for client in client_names:

        
        
        local_model = MyModel()
        
        
                #compile model
        local_model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics= metrics)
        
        #get global weights from server
        local_model.set_weights(global_weights)
        
        
        #train device on its local data for E=1
        local_model.fit(client_list[client], epochs=Epochs, verbose=0)
        
        #Claculate scale factor
        scale_factor = calculate_scale_factor(client_list, client)
        
        #scaled device_model weights
        scaled_weight = scale_device_weights(scale_factor, local_model.get_weights())
        
        
        list_Scaled_weights.append(scaled_weight)
        
    
        #clear session to free memory after each communication round
        K.clear_session()
    
    #agregating: calculate average of scaled weights
    Avg_weights = calculate_avg_weights(list_Scaled_weights)
    
    #update server model weights
    global_model.set_weights(Avg_weights)
    
    #test server model and print out metrics after each communications round
    for (x_test, y_test) in test_batch:
        global_acc, global_loss = test_model(x_test, y_test, global_model, rnd)
    
    









