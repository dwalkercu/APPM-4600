'''
A smoothing spline library which allows for the use of a cubic truncated power spline basis and smoothing spline etrainuation.
AUTHOR: Derek Walker
'''

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import kfold
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

def find_opt_lambda(x, data, k, min_lda=0, max_lda=1):
    """Returns the optimal lambda trainue using K-fold cross-validation of the data

    x - knots of the smoothing spline
    data - the data which the smoothing spline will minimize the error of
    k - the number of folds to use in K-fold cross validation
    min_lda - the minimum acceptable trainue of lambda
    max_lda - the maximum acceptable trainue of lambda
    """
    N = len(x) # number of knots
    opt_lda = min_lda
    scores = []
    folds = kfold.k_folds(x, data, k)

    for i in range(k):
        # select new folds
        last_ind = 0
        (val,train,last_ind) = kfold.select_new_folds(folds, last_ind)

        # train lambda
        train_lda = min_lda
        lda = min_lda
        min_err = np.inf
        for j in range(k-1):
            x_train = train[j,0]
            y_train = train[j,1]

            # make new smoothing spline
            x0 = np.linspace(np.min(x_train), np.max(x_train), (int)(len(x_train)+len(x_train)*0.25))
            ss = eval_smoothing_spline(x0, x_train, y_train, lda)

            # calculate MSE using the training set
            errs = np.zeros(N)
            for z in range(N):
                ss_ind = np.abs(x0 - x[z]).argmin()
                errs[z] = np.abs(ss[ss_ind] - data[z])

            mse = errs.mean()
            if mse < min_err:
                train_lda = lda
                min_err = mse

            lda = lda + (max_lda-min_lda)/(k-1)

        # test the spline on the validation set and record the score
        x_val = val[0]
        y_val = val[1]
        x0 = np.linspace(np.min(x_val), np.max(x_val), (int)(len(x_val)+len(x_val)*0.25))
        ss = eval_smoothing_spline(x0, x_val, y_val, train_lda)

        errs = np.zeros(N)
        for z in range(N):
            ss_ind = np.abs(x0 - x[z]).argmin()
            errs[z] = np.abs(ss[ss_ind] - data[z])

        mse = errs.mean()
        scores.append((mse, train_lda))
    
    # choose the best lambda
    opt_lda = np.min(scores, axis=0)[1]
    print(f"Optimal SS Lambda: {opt_lda}")

    return opt_lda