import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from numpy.linalg import pinv, matrix_transpose

def truncated_power_basis(x0, knots):
    n_points = len(x0)
    n_knots = len(knots)
    basis = np.zeros((n_points, n_knots))

    for i in range(0, n_knots):
        for j in range(n_points):
            basis[j][i] = np.max([0, (x0[j] - knots[i])**3])
    
    return basis

def second_derivative_penalty_matrix(truncated_power_basis, x0):
    basis = truncated_power_basis
    n_points = basis.shape[0]
    n_basis_fns = basis.shape[1]
    pmatrix = np.zeros((n_basis_fns, n_basis_fns))

    for i in range(n_basis_fns):
        d2ydx2i = np.gradient(np.gradient(basis[:,i], x0), x0)
        for j in range(n_basis_fns):
            d2ydx2j = np.gradient(np.gradient(basis[:,j], x0), x0)
            for k in range(n_points):
                pmatrix[i][j] += d2ydx2i[k]*d2ydx2j[k] # a discrete sum over many points -- the closest thing to an inner product without integrating
    
    return pmatrix

def eval_smoothing_spline(x0, x, data, lda=0.001):
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

def find_opt_lambda_ss(x, data, min_lda=0, max_lda=1, n=100):
    N = len(x)
    
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
        
        lda = min_lda + (max_lda - min_lda)/n

    print(out_lda)
    return out_lda