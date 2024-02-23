import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Project the data onto the 2 primary principal components

# load the eris dataset  
data = datasets.load_iris()

X = data.data  
y = data.target
# this code aims to reduce the number of features from 4 into 2 primary features 

def PCA( data,number_pc):
    
    final_eVector = []   #store the choosen maxmimum eigen vectors
    number_pc = number_pc   #number of primary components we need to keep

    # evaluate the mean of each feature 
    mean = np.mean(data, axis=0)
    X = data - mean

    # covariance matrix 
    #transpose because function needs samples as columns
    cov = np.cov(X.T)

    # eigenvalues, eigenvectors
    # e_val= [cm-yi]=0   e_vec = [cm-yi]u=0 
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # -> eigenvector transpose for easier calculations
    # sort eigenvectors
    
    eigenvectors = eigenvectors.T

    idxs = np.argsort(eigenvalues)[::-1]   #sort descending

    eigenvectors = eigenvectors[idxs]

    print ( " eigen vectors \n" , eigenvectors)
    # store first n eigenvectors
    final_eVector = eigenvectors[0 : number_pc]
    print ("choosen components \n", final_eVector)

    # return the new set of data
    return np.dot(X, final_eVector.T)

x_new= PCA(X,2)
print ( "our new data set " , x_new )

print("Shape of X:", X.shape)
print("Shape of transformed X:", x_new.shape)

x1 = x_new[:, 0]
print ( "x axis data ," , x1)
x2 = x_new[:, 1]
print ( "y axis data ," , x2)

plt.scatter(
    x1, x2, c=y,cmap=plt.cm.get_cmap("viridis", 3)
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()