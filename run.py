#Import necesseray library (numpy) and helpers
import numpy as np
from data_processing import *
from proj1_helpers import *
from implementations import *
from parameters import pen_log_reg_params


#Import train data and test data
train_data_path = 'Data/train.csv'
test_data_path = 'Data/test.csv'

y_data_train, x_data_train, ids_data_train = load_csv_data(train_data_path) 
y_data_test, x_data_test, ids_data_test = load_csv_data(test_data_path) 


#Setting constants used below for training and testing the model

#Index of the column corresponding to the pri jet number in the data 
PRI_JET_NUM_INDEX = 22 
MAX_ITERS = 500


#First step of preprocessing the data : split the data according to the unique categorical feature 

y_train_jet_num_grouped, x_train_jet_num_grouped, ids_train_jet_num_grouped = group_with_jetnum(
    y_data_train, x_data_train, ids_data_train, PRI_JET_NUM_INDEX)

y_test_jet_num_grouped, x_test_jet_num_grouped, ids_test_jet_num_grouped = group_with_jetnum(
    y_data_test, x_data_test, ids_data_test, PRI_JET_NUM_INDEX)

data_per_jet_num = []


#Train the data according to each group jet_number which is associated to the corresponding optimal parameters

jet_number = 0
for (lambda_, degree, gamma), y_train_jet_num, x_train_jet_num in \
        zip(logistic_best_params, y_train_jet_num_grouped, x_train_jet_num_grouped):
   
    print(f'PRI_jet_num: {jet_number}')
    

    #Preprocess the data
    tx,y, mean, std, correlated_columns = preprocess_train(x_train_jet_num,y_train_jet_num,degree=int(degree))
    jet_number += 1
    
    #Initial w is set to random and compute w
    initial_w = np.random.rand(tx.shape[1],)
    
    w, loss = reg_logistic_regression(
      y, tx, lambda_, initial_w, MAX_ITERS, gamma)
    print(w.shape, tx.shape)
    print(w)

    # Make sure to reset the labels to -1.
    # We changed them from -1 to 0 in order to run logistic regression
    y[y == 0] = -1
    
    data_per_jet_num.append((w, loss, correlated_columns, mean, std))
    # Calculate the predictions for each of the 4 subsets using the weights and then combine them
results = None

for (w, _, correlated_columns, mean, std), (_, degree, _), y_test_jet_num, x_test_jet_num, ids_test_jet_num in \
        zip(data_per_jet_num, logistic_best_params,
            y_test_jet_num_grouped, x_test_jet_num_grouped, ids_test_jet_num_grouped):

    x,ids = preprocess_test(x_test_jet_num, ids_test_jet_num, int(degree), correlated_columns, mean, std)
    print(w.shape, x.shape)
    pred = predict_labels(w, x)
    out = np.stack((ids, pred), axis=-1)
    results = out if results is None else np.vstack((results, out))
    
    
# Create the submission
create_csv_submission(results[:, 0], results[:, 1], 'final-submission.csv')
