import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv('training.csv',header='infer')
num_rows_training = train.count(axis = 1);
print(f'Number of rows in training set = {len(num_rows_training)}')

test = pd.read_csv('testing.csv',header='infer')
num_rows_testing = test.count(axis = 1);
print(f'Number of rows in testing set = {len(num_rows_testing)}')

train.head()


print()
print()






s_count_test = test['class'].str.contains('s').sum()
h_count_test = test['class'].str.contains('h').sum()
d_count_test = test['class'].str.contains('d').sum()
o_count_test = test['class'].str.contains('o').sum()

s_count_train = train['class'].str.contains('s').sum()
h_count_train = train['class'].str.contains('h').sum()
d_count_train = train['class'].str.contains('d').sum()
o_count_train = train['class'].str.contains('o').sum()

training_dist = pd.Series([s_count_train/len(num_rows_training),
                           d_count_train/len(num_rows_training),
                           h_count_train/len(num_rows_training),
                           o_count_train/len(num_rows_training)] ,
                           index = ['s', 'd', 'h', 'o'], name = "Class distribution for training data:").to_frame()

testing_dist = pd.Series([s_count_test/len(num_rows_testing),
                          d_count_test/len(num_rows_testing),
                          o_count_test/len(num_rows_testing),
                          h_count_test/len(num_rows_testing)],
                          index = ['s', 'd', 'o', 'h'], name = "Class distribution for test data:").to_frame()


print(training_dist)
print()
print(testing_dist)

print()
print()


Y = pd.Series(train['class'].values)
X = train.drop(labels = 'class' , axis = 1)


Y_train = pd.Series(train['class'].values)
Y_test = pd.Series(test['class'].values)
X_train = train.drop(labels = 'class' , axis = 1)
X_test = test.drop(labels = 'class' , axis = 1 )

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier();

clf = clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

score = accuracy_score(Y_test, Y_pred)

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

maxdepths = [1,2,3,4,5,6,7,8,9,10]

validationAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))
numFolds = 10
index = 0

for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(max_depth = depth, random_state = 1)
    scores = cross_val_score(clf, X_train, Y_train, cv= numFolds)
    validationAcc[index] = np.mean(scores)
 
    clf = clf.fit(X_train, Y_train)
    Y_predTest = clf.predict(X_test)
    
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    
    bestHyperparam = np.argmax(validationAcc)
    index += 1
    
#print("Best hyper param : " , maxdepths[bestHyperparam])
#print("test accuracy :" , testAcc[bestHyperparam])
    
"""
plt.plot(maxdepths, validationAcc, 'ro-', maxdepths, testAcc, 'bv--')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree')
plt.legend(['Validation', 'Testing'])
plt.ylim([0.7, 1.0])
"""
    
"""
-----------------------------------------------------------------------------------------------------------
"""
    
"""

from sklearn.neighbors import KNeighborsClassifier

numNeighbors = [1,3,5,7,10,15,20,25,30]
validationAcc = np.zeros(len(numNeighbors))
knn_acc = []
numFolds = 10
index = 0
for nn in numNeighbors:
    scores = cross_val_score(clf, X_train, Y_train, cv= numFolds)
    validationAcc[index] = np.mean(scores)

    #validationAcc[index] = np.mean(scores)


    
    clf = KNeighborsClassifier(n_neighbors=nn)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    knn_acc.append(accuracy_score(Y_test, Y_pred))
    index += 1 
    
bestHyperparam = np.argmax(knn_acc)
print('Best hyperparameter: k =', numNeighbors[bestHyperparam])
print('Test Accuracy =', knn_acc[bestHyperparam])
    
plt.plot(numNeighbors, knn_acc, 'ro-', numNeighbors, validationAcc, 'bv--')
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('k-nearest neighbors')
plt.legend(['Validation','Testing'])
plt.ylim([0.7,1.0])
"""

"""
-----------------------------------------------------------------------------------------------------------
"""


#from sklearn import linear_model

regularizers = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

logistic_acc = np.zeros(len(regularizers))
validationAcc = np.zeros(len(regularizers))
index = 0
numFolds = 10

for reg in regularizers:
    clf = linear_model.LogisticRegression(C = reg, penalty = 'l1', solver = 'liblinear', random_state=1)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    logistic_acc[index] = (accuracy_score(Y_test, Y_pred))
    scores = cross_val_score(clf, X_train, Y_train, cv= numFolds)
    validationAcc[index] = np.mean(scores)
    index += 1
    
    
    
bestHyperparam = np.argmax(logistic_acc)  


plt.plot(regularizers, validationAcc, 'ro-', regularizers, logistic_acc, 'bv--')
plt.xlabel('Regularizer (C)')
plt.ylabel('Accuracy')
plt.title('Logistic regression')
plt.legend(['Validation','Testing'])
plt.ylim([0.7,1.0])


print('Best hyperparameter, C =', regularizers[bestHyperparam])
print('Test Accuracy =', logistic_acc[bestHyperparam])









