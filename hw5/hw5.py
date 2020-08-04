import pandas as pd
import numpy as np
import math

data = pd.read_csv('ratings3.csv')
data.index = ['John' , 'Mary', 'Lee' , 'Joe' , 'Kim' , 'Bob']
print(data)



def is_number(num):
    return num == num and type(num) != str



def baseline(matrix, lambda1, lambda2, max_iter):
    
    # Initialize the model parameters w_u, w_i, and mu
    w_u = np.zeros(np.size(matrix,1))
    w_i = np.zeros(np.size(matrix,1))
    clean_matrix =  matrix[~np.isnan(matrix)]
    mu =  clean_matrix.mean()
    
    for i in range(max_iter):
        # Update the model parameters using the formula shown in class
        
        #calculate w_u
        temp = [0,0,0,0,0,0]
        for i,row in enumerate(matrix):
           # print("row: " , row)
            rating_count = 0
            for j, item in enumerate(row):
                if is_number(item):
                    rating_count += 1
                    temp[i] += item - w_i[j] - mu
            temp[i] = temp[i]/(rating_count - lambda1)
        w_u = temp
    
        #calculate w_i
        temp = [0,0,0,0,0]
        data_t = data.T   
        for i , column in enumerate(data_t.iterrows()):
            rating_count = 0
            for j, entry in enumerate(column):   
                    for rating in entry:
                        if is_number(rating):
                            rating_count += 1
                            temp[i] += rating - w_i[j] - mu
            temp[i] = temp[i] / (rating_count - lambda2)
        w_i = temp
                        
        #calculate mu again
        temp = 0
        rating_count = 0
        for i,row in enumerate(matrix):
            for j, item in enumerate(row):
                if is_number(item):
                    temp += item - w_u[i] - w_i[j]
                    rating_count += 1
    
        mu = temp/rating_count
    
        #Display the updated model parameters
        print('Iteration ', i)
        print('   w_u =', np.around(w_u , 3))
        print('   w_i =', np.around(w_i , 3))
        print('   mu  =', round(mu , 4))

    return (w_u, w_i, mu)



# Apply the model to the ratings data. Set lambda1=lambda2=0 and maxiter = 5

w_u, w_i, mu = baseline(data.values, 0, 0, 5)


print('Predicted rating for John on Harry Potter =',  round(w_u[1] + w_i[3] + mu , 3))
print('Predicted rating for Kim on Back to the Future =', round( w_u[4] + w_i[2] + mu , 3))
print('Predicted rating for Bob on Mission Impossible =',  round(w_u[5] + w_i[0] + mu , 3))








def MF(matrix, k, maxiter):
    
    # Initialize the missing ratings in matrix to 0
    matrix[np.isnan(matrix)] = 0
    print(matrix)

# =============================================================================
# 
# =============================================================================
    # Initialize the matrices U and M to 1s. Size of U is #rows x k, size of M is $columns x k
    U = np.full((matrix.shape[0] , k) , 1).astype(float)
    M = np.full((matrix.shape[1] , k) , 1).astype(float)


    for i in range(maxiter):
        
        # Update M
        for i , row in enumerate(M):
            for j , element in enumerate(row):
                M[i][j] = M[i][j] *( (matrix.T.dot(U)[i][j])  / (M.dot(U.T).dot(U)[i][j]) )

        
    
        # Update U
        for i , row in enumerate(U):
            for j , element in enumerate(row):
                U[i][j] = U[i][j] * (matrix.dot(M)[i][j]) / ( (U.dot(M.T).dot(M))[i][j]) 
        
        # Update the missing ratings in matrix with the predicted values U x M^T 
        
        
        
    return U, M




# Apply the model to the ratings data. Set k=2 and maxiter = 100

U, M = MF(data.values, 2, 1)
predicted =  U.dot(M.T)

print('Predicted rating for John on Harry Potter =', predicted[0][3])
print('Predicted rating for Kim on Back to the Future =', predicted[4][2])
print('Predicted rating for Bob on Mission Impossible =', predicted[5][0])












