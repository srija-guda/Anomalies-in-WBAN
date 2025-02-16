import os
import numpy as np
import pandas as pd
def gen_data_w(i):
    csv_files = [f'dataset/{j}.csv' for j in range(15*(i-1)+1, 15*i+1)]
    df=pd.concat([pd.read_csv(f) for f in csv_files])
    #make a new column Anamoly and assign 0 to it if it is normal and 1 if it is abnormal
    df['Anomaly'] = np.where( df["Anomaly_Type"] == 0, 0, 1)
    return df
def gen_data_custom(list):
    #the list will have the patients for whom data i want to merge
    csv_files=[]
    for j in list:
        for i in range(15*(j-1)+1, 15*j+1):
            csv_files.append(f'dataset/{i}.csv')
    df=pd.concat([pd.read_csv(f) for f in csv_files])
    df_new=df[df["Anomaly_Type"] == 0]
    return df_new['Body_Temperature'], df_new['Heart_Rate'], df_new['Pulse_Rate'], df_new['SpO2'], df_new['ECG']
def data_allp_tr():
    csv_files = [f'dataset/{j}.csv' for j in range(1, 241)]
    df=pd.concat([pd.read_csv(f) for f in csv_files])
    df_new=df[ df["Anomaly_Type"] == 0]
    return df_new['Body_Temperature'], df_new['Heart_Rate'], df_new['Pulse_Rate'], df_new['SpO2'], df_new['ECG']
def gen_data_normal(i):
    # i is patient number
    # so data for i patient is from cvs files 15*i+1 to 15*i+15
    csv_files = [f'dataset/{j}.csv' for j in range(15*(i-1)+1, 15*i+1)]
    df=pd.concat([pd.read_csv(f) for f in csv_files])
    # print(df['Normal'].value_counts())
    # df=pd.read_csv(r'dataset/'+str(i)+'.csv')
    df_new=df[ df["Anomaly_Type"] == 0]
    return df_new['Body_Temperature'], df_new['Heart_Rate'], df_new['Pulse_Rate'], df_new['SpO2'], df_new['ECG']
def gen_data_abnormal(i):
    csv_files = [f'dataset/{j}.csv' for j in range(15*(i-1)+1, 15*i+1)]
    df=pd.concat([pd.read_csv(f) for f in csv_files])
    df_new=df[df["Anomaly_Type"] != 0]
    return df_new['Body_Temperature'], df_new['Heart_Rate'], df_new['Pulse_Rate'], df_new['SpO2'], df_new['ECG']
    