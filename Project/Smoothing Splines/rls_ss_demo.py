import cubic_splines as cs
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from numpy.linalg import pinv, matrix_transpose
from scipy.interpolate import make_smoothing_spline

def discrete_least_squares(x, data, deg=1):
    N = len(data)
    M = np.zeros((N, deg+1))

    for i in range(N):
        for j in range(deg+1):
            M[i][j] = (x[i])**j

    b = data
    coefs = pinv(matrix_transpose(M) @ M) @ (matrix_transpose(M) @ b)

    return (M,coefs)

def regularized_least_squares(x, data, deg=2, lda=0.01, M=None):
    N = len(data)

    # construct M 
    if M == None:
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

    b = data
    coefs = pinv(matrix_transpose(M) @ M + lda*(matrix_transpose(gamma) @ gamma)) @ (matrix_transpose(M) @ b)

    return M @ coefs

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
    rls = regularized_least_squares(x, data, deg=3)
    ss = cs.eval_smoothing_spline(x0, x, data, lda=1e-7)
    spss = make_smoothing_spline(x, data, lam=1e-5)

    plt.scatter(x, data, label="data")
    plt.plot(x, rls, 'r-', label="rls")
    plt.plot(x0, ss, 'g-', label="my smoothing spline")
    #plt.plot(x, spss(x), 'b-', label="scipy smoothing spline")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    driver()