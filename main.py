import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import numpy as np
import pandas as pd
from gen_data import data_allp_tr, gen_data_custom, gen_data_normal, gen_data_abnormal, gen_data_w
from model_gen import LSTM_encoder_custom, build_autoencoder, simple_autoencoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def create_dataset(X, time_steps=1):
    Xs= []
    for i in range(len(X)-time_steps+1):
        v=X[i:(i+time_steps)]
        Xs.append(v)
    return np.array(Xs)


# now ham patient 1 k normal par train karenge aur fir patient 6 k pure data par anomaly detection karenge


seq_len=1
scaler=StandardScaler()
#training data prepare
list=[14]
temp_train , hr_train, pr_train, spo2_train, ecg_train = gen_data_custom(list)
# Apply SMOTE to the training data

# temp_train , hr_train, pr_train, spo2_train, ecg_train = gen_data_normal(1)
tem_train=scaler.fit_transform(np.array(temp_train).reshape(-1,1))
X_train=create_dataset(tem_train,seq_len)

#training
model=LSTM_encoder_custom(seq_len,1)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train,X_train,epochs=5,batch_size=32,validation_split=0.2,callbacks=[callback])
X_pred_train=model.predict(X_train)
mse_train=np.mean(np.power(X_train-X_pred_train,2),axis=(1,2))
threshold=np.mean(mse_train)+3*np.std(mse_train)

# testing 
# temp_test , hr_test, pr_test, spo2_test, ecg_test,anamly = gen_data_w(4)
# tem_test=scaler.transform(np.array(temp_test).reshape(-1,1))
# X_test=create_dataset(tem_test,seq_len)
# X_pred=model.predict(X_test)
# mse=np.mean(np.power(X_test-X_pred,2),axis=(1,2))
# mse=np.where(mse>threshold,1,0)
# conf=confusion_matrix(anamly,mse)
# print(conf)
# print(f'Accuracy {accuracy_score(anamly,mse)}')

# testing on all patients
accuracies=[]
for i in range(1,17):
    temp_test , hr_test, pr_test, spo2_test, ecg_test,anamly = gen_data_w(i)  
    tem_test=scaler.transform(np.array(temp_test).reshape(-1,1))
    X_test=create_dataset(tem_test,seq_len)
    X_pred=model.predict(X_test)
    mse=np.mean(np.power(X_test-X_pred,2),axis=(1,2))
    mse=np.where(mse>threshold,1,0)
    accuracies.append(accuracy_score(anamly,mse))

#printing the accuracies in tabular format in 3 decimal places
print(f"Accuraies for all patients with training on {list}")
for i in range(1,17):
    print(f'Patient {i} : {accuracies[i-1]*100:.3f}')
