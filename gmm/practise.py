# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate synthetic data
np.random.seed(0)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0)
print(X)
# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Plot the data points and their estimated clusters
plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), s=40, cmap='viridis')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=200, c='red', marker='*', label='Centroids')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Gaussian Mixture Model')
plt.show()
