import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# load iris data set

data = datasets.load_iris()
X =data.data
y =  data.target

# Project the data onto the 2 primary principal components by Minimize the Inter-Class Variability
# and Maximize the Distance Between the Mean of Classes:


def LDA(data,target, n_components):

    n_components = n_components  #number of primary components we need to keep
    eigVectors = []              #store the choosen maxmimum eigen vectors

    n_features = data.shape[1]  #store number of features 
    c_labels = np.unique(y) #get the classes of the data set in the unique form 


    mean_overall = np.mean(data, axis=0)   
    SW = np.zeros((n_features, n_features))   # declare 4*4 matrix 
    SB = np.zeros((n_features, n_features))   # declare 4*4 matrix 

    for c in c_labels:

        X_c = data[target == c]    #store only features data of the specified class
        mean_c = np.mean(X_c, axis=0)   #get the mean of data in the specified class

        #calculate within class scatter
        # SW = sum((X_class - mean_X_c)^2 )
        SW += (X_c - mean_c).T.dot((X_c - mean_c))

        nS_class = X_c.shape[0]   #number of samples in the class
        mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
        #calculate between class scatter
        # SB = sum( nS_class * (mean_X_class -total mean)^2 )
        SB += nS_class * (mean_diff).dot(mean_diff.T)  

    # Determine SW^-1 * SB

    matrix= np.linalg.inv(SW).dot(SB)

    # Get eigenvalues and eigenvectors of SW^-1 * SB

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # -> eigenvector v = [:,i] column vector, transpose for easier calculations
    # sort eigenvalues high to low

    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1] #sort descending to get the maximum 
    # get maximum eigen vectors depending on eigen values 
    eigenvectors = eigenvectors[idxs]

    # store first n eigenvectors
    eigVectors = eigenvectors[0 : n_components]

    # project data
    # return the new set of data
    return np.dot(X, eigVectors.T)


x_new =LDA(X, y,2)

print("Shape of X:", X.shape)  
print("Shape of transformed X:", x_new.shape)

x1 = x_new[:, 0]
print ( "x axis data ," , x1)
x2 = x_new[:, 1]
print ( "y axis data ," , x2)

plt.scatter(x1, x2, c=y,  alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.show()