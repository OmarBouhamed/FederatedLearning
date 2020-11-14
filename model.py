from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def MyModel():    
    model = Sequential(
        [
            keras.Input(shape=(28*28)),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10)
        ]
    )
    return model
    
    
    








