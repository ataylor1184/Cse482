import pandas as pd

#1 
data = pd.read_csv('mammography.csv', names = ["0","1","2","3","4","5","6"])



#2
import matplotlib
#%matplotlib inline

data.plot.scatter(x= '3' ,y= '4' ,c= '6',colormap='cool')

    
#3

classes = data['6']
data = data.drop(labels = ['6'], axis = 1)
print('Size of data:', data.shape)
print('Class distribution:')
print(classes.value_counts())


#4

import numpy as np

centered_data = (data.iloc[:] - data.mean()).values
inv_S = np.linalg.inv(data.cov().values)

S = data.cov()
inv_S = pd.DataFrame(np.linalg.inv(S.values),S.columns,S.index)

Z = np.zeros((centered_data.shape[0], 1))

for i in range(Z.shape[0]):
    Z[i] = np.dot(centered_data[i] , np.dot(inv_S, centered_data[i].T)) 

# Sort the Z-scores and assign the top-5 highest scores as anomalies.

outlier_list = [float(x) for x in sorted(Z, reverse = True)[0:5]]
result = pd.DataFrame()
result['Z-Score'] = [x for x in Z]
result['Prediction'] = [1 if x  in outlier_list else 0 for x in Z]
#print(result)



#5

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Accuracy =', accuracy_score(classes, result['Prediction']) )
cm = confusion_matrix(classes, result['Prediction'])
pd.DataFrame(cm)

#6

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=200, max_samples=50, contamination=0.025, 
                      random_state=1)
clf.fit(data.values)
score2 = clf.predict(data.values)
score2
result2 = pd.DataFrame()
result2['Prediction'] = [0 if x == 1 else 1 for x in score2]

print('Accuracy =', accuracy_score(classes, result2['Prediction']) )
cm2 = confusion_matrix(classes, result2['Prediction'])
pd.DataFrame(cm2)







#relabel normal points as zero
#label outliers as 1 in isolation forest