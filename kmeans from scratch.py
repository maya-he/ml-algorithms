import numpy as np 
import matplotlib.pyplot as plt

datapoints = np.array([
    [2,10],
    [2,5],
    [8,4],
    [5,8],
    [7,5],
    [6,4],
    [1,2],
    [4,9]
])

def calculate_distance(point1, point2):
    # return np.sqrt(np.sum((point1-point2)**2))
    return np.linalg.norm(point1 - point2)

def kmeans(data,k,max_iter):
    centroids = data[np.random.choice(len(data), k , replace=False)]    #random centers

    for _ in range(max_iter):
        distance_list = []
        for point in data:
            distance_from_centroid = [calculate_distance(point,center) for center in centroids] #[1.5   2.5   3]
            distance_list.append(distance_from_centroid)
            distances = np.array(distance_list)

        labels = np.argmin(distances, axis=1) 

        #iterate over each cluster
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis =0)

    return labels, centroids


labels, centroids = kmeans(datapoints,2,100)
print("\nfinal centroids")
print(centroids)
print("labels :", labels)

plt.scatter(datapoints[:,0], datapoints[:,1], c=labels , label="Data points")
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='X',s=200, label='centers')
plt.title('K-means clustering')
plt.legend()
plt.show()


              
