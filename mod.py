import pandas as pd
import numpy as np
from tensorflow import keras
input_dim = 1  

autoencoder = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),  
    keras.layers.Dense(4, activation='relu'),  
    keras.layers.Dense(8, activation='relu'),  
    keras.layers.Dense(1, activation='linear')  
])

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train_data, train_data, epochs=50, batch_size=16, shuffle=True)

test_data = pd.read_csv('10.csv')
dat = test_data[["Body_Temperature"]]

output = autoencoder.predict(dat)
out_err = tf.keras.losses.mse(output, dat)
threshold = tf.reduce_mean(out_err)
threshold

test_data["Anomaly_Score"] = out_err.numpy() 
test_data["Prediction"] = (test_data["Anomaly_Score"] > threshold.numpy()).astype(int)

TP = test_data[(test_data['Prediction'] == 1) & (test_data['Anomaly_Type'] == 1)].shape[0]
TN = test_data[(test_data['Prediction'] == 0) & (test_data['Anomaly_Type'] == 0)].shape[0]
FP = test_data[(test_data['Prediction'] == 1) & (test_data['Anomaly_Type'] == 0)].shape[0]
FN = test_data[(test_data['Prediction'] == 0) & (test_data['Anomaly_Type'] == 1)].shape[0]

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100

print(f"Accuracy: {accuracy}%")