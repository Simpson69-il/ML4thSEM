# -*- coding: utf-8 -*-
"""
EXPT NO: 7 - GMM Clustering
Name: Mohd Shafique RB
Roll No: 24BAD074
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)

gmm_labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

df['GMM_Cluster'] = gmm_labels

sil_score = silhouette_score(X_scaled, gmm_labels)
print("Silhouette Score:", sil_score)

print("Log Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))

means = gmm.means_

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels)
plt.scatter(means[:, 0], means[:, 1], s=200, marker='X')
plt.title("GMM Clustering with Means")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
# %%
plt.show()


print("\nCluster Probabilities (first 5 rows):\n", probs[:5])

print("\nCluster Summary:")
print(df.groupby('GMM_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

x = np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 200)
y = np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 200)

X_mesh, Y_mesh = np.meshgrid(x, y)

XX = np.array([X_mesh.ravel(), Y_mesh.ravel()]).T

Z = -gmm.score_samples(XX)
Z = Z.reshape(X_mesh.shape)

plt.figure(figsize=(8, 6))
plt.contour(X_mesh, Y_mesh, Z, levels=20)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, s=30)

plt.title("GMM Contour Plot")
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.grid(True)
plt.show()