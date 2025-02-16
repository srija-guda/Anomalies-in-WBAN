def validate_anomaly(sensor_idx, anomalies, correlation_matrix, threshold=0.2):
    features = ['temp_train_2', 'hr_train_2', 'pr_train_2', 'spo2_train_2', 'ecg_train_2']
    corr = []
    for i in range(5):
        if i == sensor_idx:
            continue
        if anomalies[i] == 1:
            if abs(correlation_matrix.loc[features[sensor_idx], features[i]]) > threshold:
                corr.append(i)
    for s in corr:
        if anomalies[s] == 1 :
            return True
    return False

from scipy.stats import pearsonr
import pickle

seq_len=1
scaler_temp=StandardScaler()
temp_train , hr_train, pr_train, spo2_train, ecg_train = gen_data_custom(list)
scaler_temp=StandardScaler()
temp_train = scaler_temp.fit_transform(np.array(temp_train).reshape(-1,1))
scaler_hr=StandardScaler()
hr_train=scaler_hr.fit_transform(np.array(hr_train).reshape(-1,1))
scaler_pr=StandardScaler()
pr_train=scaler_pr.fit_transform(np.array(pr_train).reshape(-1,1))
scaler_spo2=StandardScaler()
spo2_train=scaler_spo2.fit_transform(np.array(spo2_train).reshape(-1,1))
scaler_ecg=StandardScaler()
ecg_train=scaler_ecg.fit_transform(np.array(ecg_train).reshape(-1,1))
temp_train_2 = pd.Series(temp_train.flatten(), name='temp_train_2')
hr_train_2 = pd.Series(hr_train.flatten(), name='hr_train_2')
pr_train_2 = pd.Series(pr_train.flatten(), name='pr_train_2')
spo2_train_2 = pd.Series(spo2_train.flatten(), name='spo2_train_2')
ecg_train_2 = pd.Series(ecg_train.flatten(), name='ecg_train_2')
combined_df = pd.concat([temp_train_2, hr_train_2, pr_train_2, spo2_train_2, ecg_train_2], axis=1)
corr_matrix = combined_df.corr(method='pearson')
print(corr_matrix)
X_train_temp=create_dataset(temp_train,seq_len)
model_temp=LSTM_encoder_custom(seq_len,1)
X_train_hr = create_dataset(hr_train, seq_len)
model_hr = LSTM_encoder_custom(seq_len, 1)
X_train_pr = create_dataset(pr_train, seq_len)
model_pr = LSTM_encoder_custom(seq_len, 1)
X_train_spo2 = create_dataset(spo2_train, seq_len)
model_spo2 = LSTM_encoder_custom(seq_len, 1)
X_train_ecg = create_dataset(ecg_train, seq_len)
model_ecg = LSTM_encoder_custom(seq_len, 1)
# print(model_temp.summary())
# callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
# model_temp.fit(X_train,X_train,epochs=EPOCH,batch_size=32,validation_split=0.2,callbacks=[callback])
model_temp.fit(X_train_temp,X_train_temp,epochs=EPOCH,batch_size=32,validation_split=0.2,verbose=0)
model_hr.fit(X_train_hr, X_train_hr, epochs=EPOCH, batch_size=32, validation_split=0.2, verbose=0)
model_pr.fit(X_train_pr, X_train_pr, epochs=EPOCH, batch_size=32, validation_split=0.2, verbose=0)
model_spo2.fit(X_train_spo2, X_train_spo2, epochs=EPOCH, batch_size=32, validation_split=0.2, verbose=0)
model_ecg.fit(X_train_ecg, X_train_ecg, epochs=EPOCH, batch_size=32, validation_split=0.2, verbose=0)

with open ('/content/drive/MyDrive/model_temp.pkl', 'wb') as file:
    pickle.dump(model_temp, file)
with open ('/content/drive/MyDrive/model_hr.pkl', 'wb') as file:
    pickle.dump(model_hr, file)
with open ('/content/drive/MyDrive/model_pr.pkl', 'wb') as file:
    pickle.dump(model_pr, file)
with open ('/content/drive/MyDrive/model_spo2.pkl', 'wb') as file:
    pickle.dump(model_spo2, file)
with open ('/content/drive/MyDrive/model_ecg.pkl', 'wb') as file:
    pickle.dump(model_ecg, file)

import pickle
with open ('/content/drive/MyDrive/model_temp.pkl', 'rb') as file:
    model_temp = pickle.load(file)
with open ('/content/drive/MyDrive/model_hr.pkl', 'rb') as file:
    model_hr = pickle.load(file)
with open ('/content/drive/MyDrive/model_pr.pkl', 'rb') as file:
    model_pr = pickle.load(file)
with open ('/content/drive/MyDrive/model_spo2.pkl', 'rb') as file:
    model_spo2 = pickle.load(file)
with open ('/content/drive/MyDrive/model_ecg.pkl', 'rb') as file:
    model_ecg = pickle.load(file)
X_pred_train_temp=model_temp.predict(X_train_temp)
X_pred_train_hr=model_hr.predict(X_train_hr)
X_pred_train_pr=model_pr.predict(X_train_pr)
X_pred_train_spo2=model_spo2.predict(X_train_spo2)
X_pred_train_ecg=model_ecg.predict(X_train_ecg)
mse_train_temp=np.mean(np.power(X_train_temp-X_pred_train_temp,2),axis=(1,2))
mse_train_hr=np.mean(np.power(X_train_hr-X_pred_train_hr,2),axis=(1,2))
mse_train_pr=np.mean(np.power(X_train_pr-X_pred_train_pr,2),axis=(1,2))
mse_train_spo2=np.mean(np.power(X_train_spo2-X_pred_train_spo2,2),axis=(1,2))
mse_train_ecg=np.mean(np.power(X_train_ecg-X_pred_train_ecg,2),axis=(1,2))
# setting threshold to mean +3*std
threshold_temp=np.mean(mse_train_temp)+3*np.std(mse_train_temp)
threshold_hr=np.mean(mse_train_hr)+3*np.std(mse_train_hr)
threshold_pr=np.mean(mse_train_pr)+3*np.std(mse_train_pr)
threshold_spo2=np.mean(mse_train_spo2)+3*np.std(mse_train_spo2)
threshold_ecg=np.mean(mse_train_ecg)+3*np.std(mse_train_ecg)
accuracies_total=[]
for i in range(1,2):
    temp_test , hr_test, pr_test, spo2_test, ecg_test,anamly = gen_data_w(i)
    temp_test=scaler_temp.transform(np.array(temp_test).reshape(-1,1))
    hr_test=scaler_hr.transform(np.array(hr_test).reshape(-1,1))
    pr_test=scaler_pr.transform(np.array(pr_test).reshape(-1,1))
    spo2_test=scaler_spo2.transform(np.array(spo2_test).reshape(-1,1))
    ecg_test=scaler_ecg.transform(np.array(ecg_test).reshape(-1,1))
    X_test_temp=create_dataset(temp_test,seq_len)
    X_test_hr=create_dataset(hr_test,seq_len)
    X_test_pr=create_dataset(pr_test,seq_len)
    X_test_spo2=create_dataset(spo2_test,seq_len)
    X_test_ecg=create_dataset(ecg_test,seq_len)
    X_pred_test_temp=model_temp.predict(X_test_temp)
    X_pred_test_hr=model_hr.predict(X_test_hr)
    X_pred_test_pr=model_pr.predict(X_test_pr)
    X_pred_test_spo2=model_spo2.predict(X_test_spo2)
    X_pred_test_ecg=model_ecg.predict(X_test_ecg)
    mse_test_temp=np.mean(np.power(X_test_temp-X_pred_test_temp,2),axis=(1,2))
    mse_test_hr=np.mean(np.power(X_test_hr-X_pred_test_hr,2),axis=(1,2))
    mse_test_pr=np.mean(np.power(X_test_pr-X_pred_test_pr,2),axis=(1,2))
    mse_test_spo2=np.mean(np.power(X_test_spo2-X_pred_test_spo2,2),axis=(1,2))
    mse_test_ecg=np.mean(np.power(X_test_ecg-X_pred_test_ecg,2),axis=(1,2))
    mse_test = np.zeros(len(temp_test))
    c=0
    for row in range(len(temp_test)):
        lstm_an = np.zeros(5)
        if mse_test_temp[row] > threshold_temp:
            lstm_an[0]=1;
        if mse_test_hr[row] > threshold_hr:
            lstm_an[1] = 1
        if mse_test_pr[row] > threshold_pr:
            lstm_an[2] = 1
        if mse_test_spo2[row] > threshold_spo2:
            lstm_an[3] = 1
        if mse_test_ecg[row] > threshold_ecg:
            lstm_an[4] = 1
        final_anomalies = np.zeros(5)
        for i in range(5):
            if lstm_an[i] == 1:
                if validate_anomaly(i, lstm_an, corr_matrix):
                    final_anomalies[i] = 1
                else:
                    final_anomalies[i] = 0
        print("Initial LSTM Anomalies:", lstm_an)
        print("Final Validated Anomalies:", final_anomalies)
        #count=0
        for i in final_anomalies:
          if(i==1):
            #count+=1;
            #if(count>=3):
            mse_test[c]=1
            c+=1
            break;
    TP = np.sum(np.logical_and(anamly == 1, mse_test == 1))
    FP = np.sum(np.logical_and(anamly == 0, mse_test == 1))
    TN = np.sum(np.logical_and(anamly == 0, mse_test == 0))
    FN = np.sum(np.logical_and(anamly == 1, mse_test == 0))
    print(f'{TP} , {TN}, {FP}, {FN}')
    accuracies_total.append(accuracy_score(anamly,mse_test))

print(f"Accuraies for all patients with training on {list}")
for i in range(1,2):
    print(f'Patient {i} : {accuracies_total[i-1]*100:.3f}')