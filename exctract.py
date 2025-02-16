import matplotlib.pyplot as plt
import seaborn as sns
import gen_data as gd
# Assuming you have your data ready from gd.gen_data_w(10) and gd.gen_data_w(15)
temp, hr, pr, spo2, ecg, y = gd.gen_data_w(16)
temp1, hr1, pr1, spo21, ecg1, y1 = gd.gen_data_w(15)

# List of the data pairs for plotting
data_pairs =[ (temp, temp1), (hr, hr1), (pr, pr1), (spo2, spo21), (ecg, ecg1)]

# Create 5 plots, one for each pair of data++++++++++++++++++++++++++++++

for idx, (data1, data2) in enumerate(data_pairs):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1, color='red', linewidth=2, label=f'{chr(97 + idx)}1')
    sns.kdeplot(data2, color='blue', linewidth=2, label=f'{chr(97 + idx)}')
    plt.ylabel("Density")
    plt.title(f"Value of {data_pairs[idx][0].name}")
    plt.legend()
    plt.show()
