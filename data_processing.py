import numpy as np


def remove_features_constants_values(tx, undefined_value = -999): 
    'Method to remove features which have either only undefined values or constants values for all the data'
    tx = tx.copy()
    txT = np.transpose(tx)
    
    for i in reversed(range(txT.shape[0])):
        if np.all(txT[i] == txT[i][0]):
            tx = np.delete(tx, i, axis=1)
    return tx


def normalise(tx):
    mean = np.mean(tx)
    std = np.std(tx)
    return (tx - mean) / std, mean, std


#method to remove correlated columns
def remove_correlated_features(a,threshold = 0.85):
    matrix = a.copy()
    #get the number of rows and columns
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    correlated_columns = []
    #loop over the columns
    for i in range(columns):
        #get the column
        column = matrix[:,i]
        #loop over the remaining columns
        for j in range(i+1,columns):
            #get the column
            column2 = matrix[:,j]
            #get the correlation
            correlation = np.corrcoef(column,column2)[0,1]
            #if the correlation is above the threshold
            if correlation > threshold:
                #set the column to nan
                matrix[:,j] = np.nan
                correlated_columns.append(j)
    #return the matrix
    matrix = matrix[:, ~np.isnan(matrix).any(axis=0)]
    return matrix, list(set(correlated_columns))


def remove_rows_with_undefined_values(tx, value = -999):
    l = np.where(tx == value)[0]
    l = list(set(l))
    return l

def remove_list_rows(tx, y,l) :
    tx = np.delete(tx, l, axis=0)
    y = np.delete(y, l, axis = 0)
    
    return tx, y

def remove_outliers(matrix,threshold = 1.5):
    'Using the IQR method, we remove data which value for certain features have great difference compared to the value of the rest of the data for this feature'
    #get the number of rows and columns
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    l = []
    #get the quartiles
    quart = quartiles(matrix)
    #loop over the columns
    for i in range(columns):
        #get the 25th and 75th percentile
        q25 = quart[0,i]
        q75 = quart[1,i]

        #get the interquartile range
        iqr = q75 - q25

        #get the lower and upper bounds
        lower = q25 - (threshold * iqr)
        upper = q75 + (threshold * iqr)
        #loop over the rows
        for j in range(rows):
            #get the value
            value = matrix[j,i]
            #if the value is outside the bounds
            if value < lower or value > upper:
                #set the row to nan
                l.append(j)
    #return the matrix
    l = list(set(l))
    return l

#method to compute quartiles of the columns of a matrix
def quartiles(matrix):
    #get the number of rows and columns
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    #create a matrix to store the quartiles
    quartiles = np.zeros((2,columns))
    #loop over the columns
    for i in range(columns):
        #sort the column
        column = np.sort(matrix[:,i])
        #get the 25th, 50th and 75th percentile
        quartiles[0,i] = np.percentile(column,25)
        quartiles[1,i] = np.percentile(column,75)
    #return the quartiles
    return quartiles


def build_poly(tx, degree):
    poly = np.ones((tx.shape[0], 1))
    for deg in range(1, degree+1):
        poly = np.concatenate((poly, np.power(tx, deg)), axis=1)

   
    if degree != 1:
        
        pairwise = np.array([tx[:, i] * tx[:, j] for i in range(tx.shape[1])
                            for j in range(i+1, tx.shape[1])])
        poly = np.concatenate((poly, pairwise.T), axis=1)

    return poly


def preprocess_train(tx,y,degree):

    x = tx.copy()

    x = remove_features_constants_values(x)
    
    rows_undefined_values = remove_rows_with_undefined_values(x)
    
    x,y = remove_list_rows(x,y,rows_undefined_values)
    
    outliers_index = remove_outliers(x)
    
    x,y = remove_list_rows(x,y,outliers_index)
    

    x, correlated_columns = remove_correlated_features(x)

    x = build_poly(x, degree)

    # Normalise data after expanding it except the first column which is all 1
    x[:, 1:], mean, std = normalise(x[:, 1:])

    return x,y, mean, std, correlated_columns

def preprocess_test(tx,ids,degree, correlated_columns, mean, std) : 
    
    x = tx.copy()
    
    x = remove_features_constants_values(x)
    
    
    x = np.delete(x, correlated_columns, axis=1)
    
    x[x == -999] = np.nan
    
    x = build_poly(x, degree)

    x[:, 1:] = (x[:, 1:] - mean) / std
    
    x[np.isnan(x)] = 0.0
    
    return x, ids
