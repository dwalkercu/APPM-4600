'''
A smoothing spline library which allows for the use of a cubic truncated power spline basis and smoothing spline etrainuation.
AUTHOR: Derek Walker
'''

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from numpy.linalg import pinv, matrix_transpose

def truncated_power_basis(x0, knots):
    """Returns a 2D numpy array of floats containing the basis functions etrainuated at the given etrainuation points.

    x0 - etrainuation points
    knots - points which the basis functions will be defined upon -- they will be 0 to the left of each knot
    """
    n_points = len(x0)
    n_knots = len(knots)
    basis = np.zeros((n_points, n_knots))

    basis[:,0] = 1
    basis[:,1] = x0
    basis[:,2] = x0**2
    basis[:,3] = x0**3

    # discard the first two knots to have a basis the same length as the number of knots
    for i in range(n_knots): 
        for j in range(n_points):
            # calculate the truncated power basis functions
            basis[j][i] = np.max([0, (x0[j] - knots[i])**3])
    
    return basis

def second_derivative_penalty_matrix(basis, x0, knots):
    """Returns the nxn second derivative penalty matrix for the given spline basis functions

    basis - basis functions of the form mxn with m etrainuation points and n basis functions associated with n knots
    x0 - etrainuation points of the basis
    """
    n_knots = len(knots)
    n_points = basis.shape[0]
    n_basis_fns = basis.shape[1]
    pmatrix = np.zeros((n_points, n_basis_fns))

    for i in range(n_knots):
        for j in range(n_basis_fns):
            d2ydx2j = np.gradient(np.gradient(basis[:,j], x0), x0)
            # second derivative of basis function etrainuated at the ith etrainuation point
            pmatrix[i][j] = d2ydx2j[np.where(x0 >= knots[i])[0][0]]
    
    return pmatrix

def eval_smoothing_spline(x0, x, data, lda=0.001):
    """Returns the y-trainues of a smoothing spline using x0 etrainuation points with x knots
    This smoothing spline uses the truncated power basis with the second derivative penalty matrix
    The individual splines are cubic polynomials.

    x0 - etrainuation points
    x - knots
    data - the data which to minimize the error of using the smoothing spline
    lda - the tuning parameter for the second derivative penalty matrix
    """
    N = len(x)
    Netrain = len(x0)
    y = np.zeros(Netrain)
    basis = truncated_power_basis(x0, x)
    n_basis_fns = basis.shape[1]
    
    # construct G
    G = np.zeros((N, n_basis_fns))
    for i in range(N):
        node_ind = np.where(x0 >= x[i])[0][0]
        for j in range(n_basis_fns):
            G[i][j] = basis[node_ind][j] 

    # construct second-derivative penalty matrix
    gamma = second_derivative_penalty_matrix(basis, x0, x)

    # get coefs for basis functions
    b = data
    coefs = pinv(matrix_transpose(G) @ G + lda*matrix_transpose(gamma) @ gamma) @ (matrix_transpose(G) @ b)

    # construct splines
    for i in range(Netrain):
        tmp_sum = 0
        for k in range(n_basis_fns):
            tmp_sum += coefs[k]*basis[i][k]
        y[i] = tmp_sum

    return y

def find_opt_lambda(x, data, min_lda=0, max_lda=1, n=100):
    """Returns the optimal lambda trainue using standard cross-trainidation of the data

    x - knots of the smoothing spline
    data - the data which the smoothing spline will minimize the error of
    min_lda - the minimum acceptable trainue of lambda
    max_lda - the maximum acceptable trainue of lambda
    n - the size of the training set
    """
    N = len(x) # number of knots
    
    # get a subsample of the data as a training set
    train_size = int(0.20*N) if N>20 else 2
    ind = np.arange(0, N, 1)
    np.random.shuffle(ind)
    train_ind = np.random.choice(ind, train_size, replace=False)
    train_ind = np.sort(train_ind)
    train_x = x[train_ind]
    train = data[train_ind]

    # get the optimal lambda using the training set
    x0 = np.linspace(np.min(train_x), np.max(train_x), N)
    min_err = np.inf
    out_lda = min_lda
    lda = min_lda
    for _ in range(n):
        ss = eval_smoothing_spline(x0, train_x, train, lda)

        # get error at every point on the smoothing spline
        err = np.zeros(train_size)
        for i,v_ind in zip(range(train_size), train_ind):
            # get the knot x0 index to compare knots vs data points
            # this works because every data point is a knot
            ss_ind = np.where((x0 >= x[v_ind]))[0][0]
            err[i] = (train[i] - ss[ss_ind])**2

        # calculate MSE and compare to stored min_err
        mean_err = np.mean(err)
        if mean_err < min_err:
            min_err = mean_err
            out_lda = lda
        
        # update lambda
        lda = lda + (max_lda - min_lda)/n
    
    print("Optimal SS lambda: ", out_lda)

    return out_lda