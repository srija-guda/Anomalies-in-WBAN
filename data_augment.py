import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import KMeansSMOTE
import gen_data as gd

# Generate synthetic data (replace with your actual data loading)
df = gd.gen_data_w(15)

# Extract features (X) and target (y)
X = df.drop(['Anomaly_Type', 'Anomaly'], axis=1)  # Keep all features
y = df['Anomaly']  # Target variable

print(f"Original dataset size: {X.shape[0]} samples")

# Step 1: Use K-Means to find clusters in X (ignoring Y)
num_clusters = min(50, len(X))  # Ensure reasonable number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
X['Cluster'] = kmeans.fit_predict(X)  # Assign cluster labels to each sample
# Step 2: Apply KMeansSMOTE *only within clusters* to expand X
u, c = np.unique(y, return_counts=True)
print("Before resampling: ", u, c)
kmeans_smote = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=42)
X_resampled, _ = kmeans_smote.fit_resample(X.drop(columns=['Cluster']), y)
u,c=np.unique(_,return_counts=True)
print("After resampling: ",u,c)
# Print results
print(f"Resampled dataset size: {X_resampled.shape[0]} samples")
