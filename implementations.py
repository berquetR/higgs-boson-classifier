import numpy as np
from proj1_helpers import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ Computation of weights and loss using the gradient descent algorithm with least squares.
    :param y: observed data
    :param tx: input
    :param initial_w: initial weight vector
    :param max_iters: number of iterations
    :param gamma: step size for the stoch gradient descent
    :return: last weight vector, last loss
    """
    w = initial_w
    threshold = 1e-8
    losses = []

    for n_iter in range(max_iters):
        gradient = compute_least_squares_gradient(y,tx,w)
        w -= gamma * gradient

        loss = compute_mse(y, tx, w)
        losses.append(loss)

        if n_iter % 10 == 0 : 
            print(f'iteration: {n_iter} ({round(n_iter/max_iters*100, 2)}%)')
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """ Computation of weights and loss using the stochastic gradient descent algorithm.
    :param y: observed data
    :param tx: input
    :param initial_w: initial weight vector
    :param max_iters: number of iterations
    :param gamma: step size for the stoch gradient descent
    :return: last weight vector, last loss
    """
    
    threshold = 1e-8
    batch_size=1
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        # Random values of y and tx
        for y_n, x_n in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            grad = compute_least_squares_gradient(y_n, x_n, w)
            w -= grad * gamma

        loss = compute_mse(y, tx, w)
        losses.append(loss)
        
        if n_iter % 100 == 0: 
           print(n_iter/max_iters)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]


def least_squares(y, tx):
    """ Computation of weights and loss with the least squares algorithm.
    :param y: observed data
    :param tx: input
    :return: last weight vector, last loss
    """
    w, _, _, _ = np.linalg.lstsq(tx,y,rcond=None)
    loss = compute_mse(y, tx,w)

    return w, loss



def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    """ Computation of weights and loss with the least squares algorithm.
    :param y: observed data
    :param tx: input
    :param lambda_: regularization parameter
    :return: last weight vector, last loss
    """
    
    y = np.reshape(y, (len(y), 1))
    gram_matrix = tx.T @ tx

    lambda_i = 2 * lambda_ * len(y) * np.identity(len(gram_matrix))
    w = (np.linalg.inv(gram_matrix + lambda_i)) @ (tx.T @ y)
    loss = compute_mse(y,tx,w)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Computation of weights and losses using the logistic regression using gradient descent
    :param y: observed data
    :param tx: input
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size for the stoch gradient descent
    :return: last weight vector, last loss
    """

    # y in the set {-1,1} so we assign to each -1 the value 0
    y[y == -1] = 0
    w = initial_w
    threshold = 1e-8
    losses = []

    for n_iter in range(max_iters):
        gradient = tx.T @ (sigmoid(tx @ w) - y)/tx.shape[0]
        w -= gamma*gradient
        
        loss = compute_logistic_loss(y, tx, w)
        losses.append(loss)

        if n_iter % 10 == 0:
            print(f'iteration: {n_iter} ({round(n_iter/max_iters*100, 2)}%)')   
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    """Reguralized logistic regression using gradient descent
    :param y: observed data
    :param tx: input
    :param lambda_: regularization parameter
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size for the stoch gradient descent
    :return: last weight vector, last loss
    """

    # y in the set {-1,1} so we assign to each -1 the value 0
    y[y == -1] = 0
    w = initial_w
    threshold = 1e-8
    losses = []

    for n_iter in range(max_iters):
        gradient = compute_negative_log_likelihood_gradient(y, tx, w, lambda_)
        w -= gamma*gradient

        loss = compute_logistic_loss(y, tx, w)
        losses.append(loss)

        if n_iter % 10 == 0:
            print(f'iteration: {n_iter} ({round(n_iter/max_iters*100, 2)}%)')
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
