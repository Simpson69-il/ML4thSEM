
# EXPT NO: 7
print("Name: Mohd Shafique RB")
print("Roll no: 24BAD074")

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Preprocessing (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Elbow Method
# ---------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# ---------------------------
# Apply K-Means (Choose K=5 typically)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add labels to dataset
df['Cluster'] = labels


# Evaluation

sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score)


# Visualization
centroids = kmeans.cluster_centers_

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X')
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()


# Cluster Centers

centroids = kmeans.cluster_centers_
print("Centroids (scaled):\n", centroids)


# Interpretation

print("\nCluster Summary:")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())