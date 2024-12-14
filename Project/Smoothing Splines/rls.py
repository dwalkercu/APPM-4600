'''
A regularized least squares library which allows for variable degree polynomial regression as well as cross-validation.
AUTHOR: Derek Walker
'''

import smooth_splines as cs
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from numpy.linalg import pinv, matrix_transpose

def regularized_least_squares(x, data, deg=3, lda=0.01, M=None):
    """Returns the RLS polynomial used in the regression of the data input parameter.

    x - the x-values of the data
    data - the data to perform the regression on
    deg - the degree of the RLS polynomial
    lda - the tuning parameter for the penalty matrix
    M - a specified basis. None => this function will create a M basis matrix
    """
    N = len(data)

    # construct M 
    if type(M) == type(None):
        M = np.zeros((N, deg+1))
        for i in range(N):
            for j in range(deg+1):
                M[i][j] = (x[i])**j

    # construct gamma
    gamma = np.zeros((N, deg+1))
    
    # second derivative penalty matrix
    for i in range(N):
        for j in range(deg+1):
            f = lambda xvar: xvar**j # assuming canonical basis
            h = x[i+1]-x[i] if i < N-1 else x[i]-x[i-1]
            h = h if h != 0 else 1e-10
            gamma[i][j] = nd.Derivative(f, h, n=2)(x[i])

    # solve the Mx = b equation using pseudo-inverse (which uses SVD to compute the inverse)
    b = data
    coefs = pinv(matrix_transpose(M) @ M + lda*(matrix_transpose(gamma) @ gamma)) @ (matrix_transpose(M) @ b)

    return M @ coefs

def find_opt_lambda(x, data, deg=3, M=None, min_lda=1e-5, max_lda=1, n=100):
    """Returns the optimal lambda value using cross-validation -- the minimization of the error over n iterations.
    
    x - the x-values of the data
    data - the data to perform the regression on
    deg - the degree of polynomial to perform the cross-validation with
    min_lda - the minimum acceptable value of the tuning parameter lambda
    max_lda - the maximum acceptable value of the tuning parameter lambda
    n - the number of iterations to perform the cross-validation
    """
    N = len(x)
    lda = 0
    
    # get a subsample of the data as a validation set
    val_size = int(N/4) if N>8 else 2
    ind = np.arange(0, N, 1)
    np.random.shuffle(ind)
    val_ind = np.random.choice(ind, val_size, replace=False)
    val_x = x[val_ind]
    val = data[val_ind]

    # get the optimal lambda using the validation set
    min_err = np.inf
    out_lda = min_lda
    for i in range(1,n):
        if type(M) == type(None):
            rls = regularized_least_squares(val_x, val, deg=deg, lda=lda)
        else:
            rls = regularized_least_squares(val_x, val, deg=deg, lda=lda, M=M)

        # minimize error
        err = (val - rls)**2
        mean_err = np.mean(err)
        if mean_err < min_err:
            min_err = mean_err
            out_lda = lda
        
        lda = lda + (max_lda - min_lda)/n

    return out_lda 