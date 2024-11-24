import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.interpolate import CubicSpline

def create_natural_spline_matrix(x):
    N = len(x)

    # make h array
    h = np.zeros(N)
    for i in range(N-1):
        h[i] = x[i+1] - x[i]
    
    h[-1] = h[-2] # !!! stopgap !!!

    # make natural spline matrix
    A = np.zeros((N, N))

    # assert natural spline conditions 
    A[0][0] = 1
    A[N-1][N-1] = 1

    for i in range(1, N-1):
        A[i][i-1] = h[i]/6
        A[i][i] = (2/3)*(h[i] + h[i+1])
        A[i][i+1] = h[i+1]/6

    return (A,h)

def create_f_vector(f, h):
    f_vec = np.zeros(len(h))
    for i in range(1, len(h)-2):
        f_vec[i] = f(i+1)/h[i+1] - f(i)/h[i]
    return f_vec

def get_cubic_spline_coefs(f, x):
    (A,h) = create_natural_spline_matrix(x)
    f_vec = create_f_vector(f, h)
    b = inv(A) @ f_vec
    d = f(x)

    N = len(x)
    a = np.zeros(N)
    c = np.zeros(N)
    for i in range(N-1):
        a[i] = f(x[i+1])/h[i] - (h[i]/6)*b[i+1]
        c[i] = f(x[i])/h[i] - (h[i]/6)*b[i]
        #a[i] = (b[i+1]-b[i])/(6*h[i])
        #c[i] = b[i]/2

    return (a,b,c,d)

def eval_cubic_spline(x0, x, a, b, c, d):
    N = len(x)
    y = []
    pts = []

    for i in range(N-1):
        pts = x0[np.where((x[i] <= x0) & (x[i+1] >= x0))]
        yi = a[i]*(pts - x[i])**3 + b[i]*(pts - x[i])**2 + c[i]*(pts - x[i]) + d[i]
        y.extend(yi)
    
    # add final points
    pts = x0[np.where((x0 >= x[i+1]))]
    yi = a[i]*(pts - x[i])**3 + b[i]*(pts - x[i])**2 + c[i]*(pts - x[i]) + d[i]
    y.extend(yi)

    return y

def driver():

    ''' LOOK AT CORONA'S NOTES FOR EVALUATING THE SPLINE & COEFS. SOMETHING'S WRONG '''

    f = lambda x: np.cos(x)
    x0 = np.linspace(-np.pi, np.pi, 1000)
    x = np.arange(-np.pi,np.pi,0.1)

    (a,b,c,d) = get_cubic_spline_coefs(f, x)
    y = eval_cubic_spline(x0, x, a, b, c, d)

    plt.plot(x0, f(x0), label="function")
    plt.plot(x0, y, label="me")
    plt.plot(x0, CubicSpline(x, f(x))(x0), label="scipy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    driver()