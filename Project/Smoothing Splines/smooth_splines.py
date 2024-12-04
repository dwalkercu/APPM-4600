'''
A smoothing spline library which allows for the use of a truncated power spline basis and cubic smoothing spline evaluation.
AUTHOR: Derek Walker
'''

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from numpy.linalg import pinv, matrix_transpose

def truncated_power_basis(x0, knots):
    """Returns a 2D numpy array of floats containing the basis functions evaluated at the given evaluation points.

    x0 - evaluation points
    knots - points which the basis functions will be defined upon -- they will be 0 to the left of each knot
    """
    n_points = len(x0)
    n_knots = len(knots)
    basis = np.zeros((n_points, n_knots))

    basis[:,0] = 1 # constant basis function
    basis[:,1] = x0 # linear basis function

    # discard the first two knots to have a basis the same length as the number of knots
    for i in range(2, n_knots): 
        for j in range(n_points):
            # calculate the truncated basis functions
            basis[j][i] = np.max([0, (x0[j] - knots[i])**3])
    
    return basis

def second_derivative_penalty_matrix(basis, x0):
    """Returns the nxn second derivative penalty matrix for the given spline basis functions

    basis - basis functions of the form mxn with m evaluation points and n basis functions associated with n knots
    x0 - evaluation points of the basis
    """
    n_points = basis.shape[0]
    n_basis_fns = basis.shape[1]
    pmatrix = np.zeros((n_basis_fns, n_basis_fns))

    for i in range(n_basis_fns):
        # second derivative of first basis function
        d2ydx2i = np.gradient(np.gradient(basis[:,i], x0), x0)
        for j in range(n_basis_fns):
            # second derivative of second basis function
            d2ydx2j = np.gradient(np.gradient(basis[:,j], x0), x0)
            for k in range(n_points):
                # calculate the inner product using discrete sum
                pmatrix[i][j] += d2ydx2i[k]*d2ydx2j[k]
    
    return pmatrix

def eval_smoothing_spline(x0, x, data, lda=0.001):
    """Returns the y-values of a smoothing spline using x0 evaluation points with x knots
    This smoothing spline uses the truncated power basis with the second derivative penalty matrix
    The individual splines are cubic polynomials.

    x0 - evaluation points
    x - knots
    data - the data which to minimize the error of using the smoothing spline
    lda - the tuning parameter for the second derivative penalty matrix
    """
    N = len(x)
    Neval = len(x0)
    y = np.zeros(Neval)
    basis = truncated_power_basis(x0, x)
    n_basis_fns = basis.shape[1]
    
    # construct G
    G = np.zeros((N, n_basis_fns))
    for i in range(N-1):
        node_ind = np.where(x0 >= x[i])[0][0]
        for j in range(n_basis_fns):
            G[i][j] = basis[node_ind][j] 

    # fill in last row of G
    node_ind = np.where(x0 >= x[-1])[0][0]
    for j in range(n_basis_fns):
        G[-1][j] = basis[node_ind][j]

    # construct second-derivative penalty matrix
    omega = second_derivative_penalty_matrix(basis, x0)

    # get coefs for basis functions
    b = data
    coefs = pinv(matrix_transpose(G) @ G + lda*omega) @ (matrix_transpose(G) @ b)

    # construct splines
    for i in range(Neval):
        tmp_sum = 0
        for k in range(n_basis_fns):
            tmp_sum += coefs[k]*basis[i][k]
        y[i] = tmp_sum

    return y

def find_opt_lambda(x, data, min_lda=0, max_lda=1, n=100):
    """Returns the optimal lambda value using standard cross-validation of the data

    x - knots of the smoothing spline
    data - the data which the smoothing spline will minimize the error of
    min_lda - the minimum acceptable value of lambda
    max_lda - the maximum acceptable value of lambda
    n - the size of the validation set
    """
    N = len(x) # number of knots
    
    # get a subsample of the data as a validation set
    val_size = int(0.20*N) if N>20 else 2
    ind = np.arange(0, N, 1)
    np.random.shuffle(ind)
    val_ind = np.random.choice(ind, val_size, replace=False)
    val_x = x[val_ind]
    val = data[val_ind]

    # get the optimal lambda using the validation set
    x0 = np.linspace(np.min(val_x), np.max(val_x), int(N+0.005*N))
    min_err = np.inf
    out_lda = min_lda
    lda = min_lda
    for _ in range(1,n):
        ss = eval_smoothing_spline(x0, val_x, val, lda)

        # minimize error
        err = np.zeros(val_size)
        for i,v_ind,v in zip(range(val_size), val_ind, val):
            err[i] = (v - ss[v_ind])**2
        mean_err = np.mean(err)
        if mean_err < min_err:
            min_err = mean_err
            out_lda = lda
        
        # update lambda
        lda = min_lda + (max_lda - min_lda)/n

    return out_lda