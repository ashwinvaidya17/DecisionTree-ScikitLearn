from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

import numpy as np

le = preprocessing.LabelEncoder()
clf = DecisionTreeClassifier()

training = np.array([
    [3, "yes", 62, "accept"],
    [4, "yes", 70, "accept"],
    [2, "yes", 71, "reject"],
    [5, "yes", 58, "reject"],
    [1, "no", 76, "reject"],
    [6, "no", 64, "reject"],
    [2, "yes", 74, "reject"],
    [3, "yes", 75, "accept"],
    [4, "yes", 67, "accept"],
    [2, "no", 73, "reject"],
])

training[:,1] = le.fit_transform(training[:,1]) 


X = training[:,:-1]
y = training[:,-1]

clf.fit(X,y)

# test data
test = np.array([
	[3, "yes", 63],
	[1, "no", 59],
	])
test[:,1] = le.transform(test[:,1])

print(clf.predict(test))

