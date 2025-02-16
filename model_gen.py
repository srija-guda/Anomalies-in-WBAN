import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout
def build_autoencoder(seq_length, num_features):
    inputs = Input(shape=(seq_length, num_features))
    encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(num_features))(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def LSTM_encoder_custom(seq_length, num_features):
    inputs = Input(shape=(seq_length, num_features))
    encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
    encoded=Dropout(0.1)(encoded)
    encoded = LSTM(16, activation='relu', return_sequences=True)(inputs)
    encoded=Dropout(0.2)(encoded)
    encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded=Dropout(0.2)(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(num_features))(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model
    
def simple_autoencoder():
    autoencoder = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),  
    keras.layers.Dense(4, activation='relu'),  
    keras.layers.Dense(8, activation='relu'),  
    keras.layers.Dense(1, activation='linear')  
    ])
    return autoencoder
