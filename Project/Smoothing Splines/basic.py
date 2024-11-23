import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def create_natural_spline_matrix(x):
    N = len(x)

    # make h array
    h = np.zeros(N)
    for i in range(N-1):
        h[i] = x[i+1] - x[i]

    # make natural spline matrix
    A = np.zeros((N, N))

    # assert natural spline conditions 
    A[0][0] = 1
    A[N-1][N-1] = 1

    for i in range(1, N-2):
        A[i][i-1] = h[i]/6
        A[i][i] = (2/3)*(h[i] + h[i+1])
        A[i][i+1] = h[i+1]/6

    return (A,h)

def create_f_vector(f, h):
    f_vec = np.zeros(len(h))
    for i in range(len(h)-1):
        f_vec[i] = f(i+1)/h[i+1] - f(i)/h[i]
    return f_vec

def get_cubic_spline_coefs(f, x):
    (A,h) = create_natural_spline_matrix(x)
    f_vec = create_f_vector(f, h)
    b = inv(A) @ f_vec
    return (b)

def eval_cubic_spline(x0, x, a, b, c, d):
    N = len(x)
    y = np.zeros(N)
    for i in range(N):
        y[i] = a[i]*(x0 - x[i])**3 + b[i]*(x0 - x[i])**2 + c[i]*(x0 - x[i]) + d[i]
    return y

def smoothing_spline():
    pass

def driver():
    f = lambda x: x**2
    print(get_cubic_spline_coefs(f, np.arange(0,10,1)))

if __name__ == "__main__":
    driver()