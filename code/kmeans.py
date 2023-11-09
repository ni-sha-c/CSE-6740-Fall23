# Make two clusters of data and plot them
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


# Make two clusters
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
# Plot the data
plt.scatter(X[:,0], X[:,1])
plt.show()

# Create a kmeans model on our data, using 2 clusters.  random_state helps ensure that the algorithm returns the same results each time.
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
plt.show()
