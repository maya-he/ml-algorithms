from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
x = data.data
y = data.target 

X_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lda = LinearDiscriminantAnalysis()

x_transformed = lda.fit_transform(X_train,y_train)
# print(x_transformed)
plt.scatter(x_transformed[:,0],x_transformed[:,1],c=y_train)
plt.xlabel("first LDA")
plt.ylabel("second LDA")
# plt.show()

accuracy = lda.score(X_train,y_train)
print('accuracy of train set',accuracy)

x_test_transformed = lda.transform(x_test)
accuracy2 = lda.score(x_test,y_test)
print('accuracy of test set',accuracy2)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred =knn.predict(x_test)
print(y_pred)

