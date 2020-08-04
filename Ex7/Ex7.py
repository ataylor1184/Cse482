import pandas as pd

data = pd.read_csv('PollutionData.csv',header='infer')
#print(data.head())





data = data.drop(labels = ['No','cbwd' , 'year'] , axis = 1)
data = data.dropna()
#print(data.head())





Y = pd.Series(data['pm2.5'].values)
X = data.drop(labels = 'pm2.5' , axis = 1)
#print(X.head())




from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = .3 , train_size = .7 , random_state=1)


# Create linear regression object
regr = linear_model.LinearRegression()


# Fit regression model to the training set
regr.fit(X_train, y_train)



# Apply model to the test set
y_pred_test = regr.predict(X_test)


# Evaluate the results
print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
print('R-squared = %.4f' % r2_score(y_test, y_pred_test))
print('Slope = ', regr.coef_)
print('Intercept = ', regr.intercept_)
coefs = regr.coef_

plt.scatter(y_test, y_pred_test, color='blue')


