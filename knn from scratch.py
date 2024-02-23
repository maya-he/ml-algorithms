from collections import Counter
from math import dist
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x_data= iris.data    #features
y_target=iris.target   #types of flowers

X_train, X_test, y_train, y_test = train_test_split(x_data, y_target, random_state = 49, test_size = 0.2)
# print (len(X_train) ) .. 120
# print( len(X_test)) ..30

def predict(k, X_test , X_train ,y_train):
    
    final_class = []

    for i in range(len(X_test)):
        distances = []
        classes= []

        for j in range(len(X_train)):
            
            distt = dist(X_train[j],X_test[i])
            distances.append([distt, j])

        distances.sort()
        distances = distances[0:k]
        print (distances)

        for d, c in distances:
            classes.append(y_train[c])

        print ("least distance occures in which classes \n",classes)    
        print ( Counter(classes))
        ans = Counter(classes).most_common(1)[0][0]
        print ("ans \n", ans )
        final_class.append(ans)
        
    return final_class

def score( X_test, y_test):

    predictions = predict(3, X_test , X_train ,y_train)
    score =(predictions == y_test).sum() / len(y_test)
    
    return (f"{score*100}%")


prediction = predict(3, X_test , X_train ,y_train)
print (prediction == y_test)
print (score(X_test, y_test))