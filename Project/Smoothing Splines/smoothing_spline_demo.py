import cubic_splines as cs
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from numpy.linalg import inv, matrix_transpose
from scipy.interpolate import make_smoothing_spline

def discrete_least_squares(x, data, deg=1):
    N = len(data)
    M = np.zeros((N, deg+1))

    for i in range(N):
        for j in range(deg+1):
            M[i][j] = (x[i])**j

    b = data
    coefs = inv(matrix_transpose(M) @ M) @ (matrix_transpose(M) @ b)

    return (M,coefs)

def regularized_least_squares(x, data, M=None, deg=1, lda=0.5, penalty="ridge"):
    N = len(data)

    # construct M 
    if M == None:
        M = np.zeros((N, deg+1))
        for i in range(N):
            for j in range(deg+1):
                M[i][j] = (x[i])**j

    # construct gamma
    gamma = np.zeros((N, deg+1))
    
    # second derivative matrix
    if(penalty.lower() == "derivative"):
        for i in range(N):
            for j in range(deg+1):
                f = lambda xvar: xvar**j # assuming canonical basis
                h = x[i+1]-x[i] if i < N-1 else x[i]-x[i-1]
                gamma[i][j] = nd.Derivative(f, h, n=2)(x[i])

    # default to ridge regression
    else:
        for i in range(N):
            gamma[i][i] = 1

    b = data
    coefs = inv(matrix_transpose(M) @ M + lda*(matrix_transpose(gamma) @ gamma)) @ (matrix_transpose(M) @ b)

    return (M,coefs)

def eval_smoothing_splines(x0, x, data, lda=0.1):
    N = len(x)
    Neval = len(x0)
    y = np.zeros(Neval)
    basis = cs.natural_truncated_power_basis(x0, x)
    n_basis_fns = basis.shape[1]
    
    # construct G
    G = np.zeros((N, n_basis_fns))
    for i in range(N-1):
        node_ind = np.where((x0 >= x[i]))[0][0]
        for j in range(n_basis_fns):
            G[i][j] = basis[node_ind][j] 

    # fill in last row of G
    node_ind = np.where(x0 >= x[-1])[0][0]
    for j in range(n_basis_fns):
        G[-1][j] = basis[node_ind][j] 

    # construct second-derivative penalty matrix
    omega = cs.second_derivative_penalty_matrix(basis, x0)

    # get coefs for basis functions
    b = data
    coefs = inv(matrix_transpose(G) @ G + lda*omega) @ (matrix_transpose(G) @ b)

    # construct splines
    for i in range(Neval):
        tmp_sum = 0
        for k in range(n_basis_fns):
            tmp_sum += coefs[k]*basis[i][k]
        y[i] = tmp_sum

    return y

def driver():
    # generate Gaussian noise
    noise_stdev = 0.4
    noise_mean = 0
    noise_numsamples = 100
    noise = np.random.normal(noise_mean, noise_stdev, noise_numsamples)

    x = np.linspace(-1, 1, noise_numsamples)
    x0 = np.linspace(-1, 1, 1000)
    nodes = np.linspace(-1, 1, 10)

    # apply noise and generate least squares / smoothing splines solutions
    data = x**3 + noise
    (M,coefs) = regularized_least_squares(x, data, deg=3, lda=0.01, penalty="derivative")
    rls = M @ coefs
    ss = eval_smoothing_splines(x0, x, data, lda=1e-7)
    spss = make_smoothing_spline(x, data, lam=1e-5)

    plt.scatter(x, data, label="data")
    plt.plot(x, rls, 'r-', label="rls")
    plt.plot(x0, ss, 'g-', label="my smoothing spline")
    #plt.plot(x, spss(x), 'b-', label="scipy smoothing spline")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    driver()