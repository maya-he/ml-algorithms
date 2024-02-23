from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_iris()
# print(data)
x = data.data    #features
print(x)
y = data.target 

s = StandardScaler()
x_scaled = s.fit_transform(x)
print(x_scaled)

pca = PCA(n_components=2)
x_pca =pca.fit_transform(x_scaled)
# print(x_pca)

plt.scatter(x_pca[:,0], x_pca[:,1], c=y)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2 ')
plt.title('pca on iris data')
plt.show()


