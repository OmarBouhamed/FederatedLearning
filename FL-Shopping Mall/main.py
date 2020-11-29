import os
import datetime
import random


import numpy as np
from numpy import nan

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from utils import *


seed_value= 3
os.environ['PYTHONHASHSEED']=str(seed_value)


data = pd.read_csv('Shopp_mall_cleaned_data.csv', parse_dates = True, index_col = 'dttm_utc', low_memory = False)
#instead of every 15-minutes, transform data to points per day
data_d = data.resample('D').sum()
data_h = data.resample('H').sum()



#plotting one of the malls consumption
fig, ax = plt.subplots(figsize=(32,12))

plt.subplot(3, 1, 1)
plt.plot(data['value.3'])
plt.title('consuption per 15-min')

plt.subplot(3, 1, 2)
plt.plot(data_h['value.3'])
plt.title('consuption per hour')

plt.subplot(3, 1, 3)
plt.plot(data_d['value.3'])
plt.title('consuption per day')


plt.show()
fig.tight_layout()

#creating training & testing dictionaries
training = data_h.loc[:'2012-10-01', :]

testing = data_h.loc['2012-10-01':, :]

# Scaling data
scales_list = dict()

train_list = dict()
for col in training.columns:
    train_scale = MinMaxScaler()
    tra = train_scale.fit_transform(training[col].values.reshape(-1, 1))
    train_list[col] = tra
    scales_list[col] = train_scale
    
    

test_list = dict()
for col in testing.columns:
    tra = train_scale.transform(testing[col].values.reshape(-1, 1))
    test_list[col] = tra
    
    

#creating training data with window size n=12 and step=1 
x_train_dict, y_train_dict = dict(), dict()

wind = 12

for name, data in train_list.items():
    x, y = list(), list()
    for i in range(wind, len(data)-wind):
        x.append(data[i-wind:i])
        y.append(data[i])
    x_train_dict[name] = np.array(x)
    y_train_dict[name] = np.array(y)
    
    

#creating training data with window size n
x_test_dict, y_test_dict = dict(), dict()

for name, data in test_list.items():
    x, y = list(), list()
    for i in range(wind, len(data)-wind):
        x.append(data[i-wind:i])
        y.append(data[i])
    x_test_dict[name] = np.array(x)
    y_test_dict[name] = np.array(y)
    


#Create global model for the server
global_model = MyModel()

#Specify model compile settings:
loss=tf.losses.MeanSquaredError()
optimizer=tf.optimizers.Adam()
metrics=[tf.metrics.MeanAbsoluteError()]


#Federated Learning process
num_clients = len(x_train_dict)  #8
clients_per_rnd = num_clients
rounds = 10
Epochs = 5

all_clients_names = list(x_train_dict.keys())
#for each round of communication
for rnd in range(rounds):
    print('round {} started:'.format(rnd))
    #to initialize the local models
    global_weights = global_model.get_weights()
    
    #shuffle local devices: for randomness
    random.shuffle(all_clients_names)
    client_names = all_clients_names[:clients_per_rnd]
    print(client_names)
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
        local_model.fit(x_train_dict[client], y_train_dict[client], epochs=Epochs, batch_size=32)
        
        #Claculate scale factor
        scale_factor = calculate_scale_factor(x_train_dict, client)
        
        #scaled device_model weights
        scaled_weight = scale_device_weights(scale_factor, local_model.get_weights())
        
        
        list_Scaled_weights.append(scaled_weight)
        
    
        #clear session to free memory after each communication round
        K.clear_session()
    
    #agregating: calculate average of scaled weights
    Avg_weights = calculate_avg_weights(list_Scaled_weights)
    
    #update server model weights
    global_model.set_weights(Avg_weights)
    
    

#testing phase
rmse_scores = list()

fig, ax = plt.subplots(figsize=(32,24))
i=0

for name, data in x_test_dict.items():
    i+=1
    y_pred = global_model.predict(data)
    
    if name=='value.8':
        break
    
    scale= scales_list[name].scale_[0]
    inv_scale = 1/scale
    y_pred = y_pred*inv_scale
    y_test = y_test_dict[name]*inv_scale
    
    rmse_scores.append(evaluate_model(y_test, y_pred))
    
    plt.subplot(len(x_test_dict), 1, i)
    
    plt.plot(y_test[:100], marker='.',label='Target', c='#2ca02c')
    plt.plot(y_pred[:100], marker='X', label='Predictions',
                  c='#ff7f0e')
    plt.title('Elec consumption for a shopping mall {}'.format(i))
    plt.xlabel('Time(h)')
    plt.ylabel('Elec consump (khw)')
    plt.legend()


plt.show()
print('RMSE scores for future predictions (for the malls used in training): {}'.format(rmse_scores))
    
    
    
fig, ax = plt.subplots(figsize=(12,5))
y_pred = global_model.predict(x_test_dict['value.8'])
print('RMSE score for future predictions (for the mall not used in training): {}'.format(evaluate_model(y_test_dict['value.8'], y_pred))) 

scale= scales_list['value.8'].scale_[0]
inv_scale = 1/scale
y_pred = y_pred*inv_scale
y_test = y_test_dict['value.8']*inv_scale

#rmse_scores.append(evaluate_model(y_test, y_pred))

plt.plot(y_test[:100], marker='.',label='Target', c='#2ca02c')
plt.plot(y_pred[:100], marker='X', label='Predictions',
              c='#ff7f0e')
plt.title('Elec consumption for a shopping mall {}'.format(i))
plt.xlabel('Time(h)')
plt.ylabel('Elec consump (khw)')
plt.legend()


plt.show()    
    
print('RMSE score for future predictions (for the mall not used in training): {}'.format(evaluate_model(y_test, y_pred)))    
    










