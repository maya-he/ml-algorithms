import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
data = iris.data
target = iris.target

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
predicted_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(centroids)

# Visualize the data points and k-means clusters
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200,label='Centroids')
plt.title('K-Means Clustering on Iris Data')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()


