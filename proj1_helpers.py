# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(y.shape[0])
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# ************************************************************************************************
# Helper functions for implementation.py *********************************************************
# ************************************************************************************************

def sigmoid(t):
    """Sigmoid function

    : param t:        Given variable
    : return sigmoid:  The value of sigmoid function given variable t
    """

    return 1/(1+np.exp(-t))


def compute_mse(y, tx, w):
    """Calculate the mean square error.

    : param y:     Observed data (Vector: Nx1)
    : param tx:    Input data (Matrix: NxD)
    : param w:     Weigths (Vector: Dx1)
    : param N:     Number of datapoints
    : param e:     Error (Vector: Nx1)
    : return mse:   Mean square error (Scalar)
    """
    y = np.reshape(y,(len(y),1))
    N = len(y)

    # Loss by MSE (Mean Square Error)
    e = y - tx@w
    mse = 1/2*np.mean(e**2)
    return mse


def compute_least_squares_gradient(y, tx, w):
    """Compute the gradient.

    : param y:     Observed data (Vector: Nx1)
    : param tx:    Input data (Matrix: NxD)
    : param w:     Weights (Vector: Dx1)
    : param N:     Number of datapoints
    : param e:     Error (Vector: Nx1)
    : return gradient:    Gradient (Vector: Dx1)
    """
    y = np.reshape(y, (len(y), 1))
    N = len(y)
    e = y-tx@w
    gradient = -(1/N)*tx.T@e
    return gradient


def compute_logistic_loss(y, tx, w):
    """Compute the loss of a logistic regression model.

    : param y:           Observed data (Vector: Nx1)
    : param tx:          Input data (Matrix: NxD)
    : param w:           Weights (Vector: Dx1)
    : return loss:        Loss for given w
    """

    loss = np.sum(np.log(np.exp(tx @ w) + 1) - y * (tx @ w))/tx.shape[0]
    return loss


def compute_negative_log_likelihood_gradient(y, tx, w, lambda_):
    """Compute a negative log likelihood gradient

    : param y:           Observed data (Vector: Nx1)
    : param tx:          Input data (Matrix: NxD)
    : param w:           Weights (Vector: Dx1)
    : return gradient:    Gradient (Vector: Dx1)
    """
    
    gradient = tx.T @ (sigmoid(tx @ w) - y)/tx.shape[0]  + 2 * lambda_ * w
    return gradient
