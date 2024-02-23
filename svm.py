# from sklearn import svm 
# import numpy as np 

# x = np.array([
#     [2,10],
#     [2,5],
#     [8,4],
#     [5,8],
#     [7,5]
# ])

# y = [0,1,1,0,1]

# clf = svm.SVC(kernel='linear').fit(x,y)

# prediction = clf.predict([[6,7]])
# print(prediction)





from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='rbf', gamma="auto").fit(X,y)

y_pred = clf.predict(X_test)
print(y_pred)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
















