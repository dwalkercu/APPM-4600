import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import integrate

def create_natural_spline_matrix(x):
    N = len(x)

    # make h array
    h = np.zeros(N)
    for i in range(N-1):
        h[i] = x[i+1] - x[i]
    
    h[-1] = h[-2] # stopgap

    # make natural spline matrix
    A = np.zeros((N, N))

    # assert natural spline conditions 
    A[0][0] = 1
    A[N-1][N-1] = 1

    for i in range(1, N-1):
        A[i][i-1] = h[i]
        A[i][i] = 2*(h[i] + h[i+1])
        A[i][i+1] = h[i+1]

    return (A,h)

def create_f_vector(f, h, x):
    f_vec = np.zeros(len(h))
    for i in range(1, len(h)-2):
        f_vec[i] = (3/h[i+1])*(f(x[i+2]) - f(x[i+1])) - (3/h[i])*(f(x[i+1]) - f(x[i]))
    return f_vec

def get_cubic_spline_coefs(f, x):
    (A,h) = create_natural_spline_matrix(x)
    f_vec = create_f_vector(f, h, x)
    b = inv(A) @ f_vec
    d = f(x)

    N = len(x)
    a = np.zeros(N)
    c = np.zeros(N)
    for i in range(N-1):
        a[i] = (b[i+1] - b[i])/(3*h[i])
        c[i] = (f(x[i+1]) - f(x[i]))/(h[i]) - (h[i]/3)*(2*b[i] + b[i+1])

    return (a,b,c,d)

def eval_cubic_splines(x0, x, a, b, c, d):
    N = len(x)
    y = []
    pts = []

    for i in range(N-1):
        pts = x0[np.where((x[i] <= x0) & (x[i+1] >= x0))]
        yi = a[i]*(pts - x[i])**3 + b[i]*(pts - x[i])**2 + c[i]*(pts - x[i]) + d[i]
        y.extend(yi)
    
    # add final points
    i += 1
    pts = x0[np.where((x0 >= x[i]))]
    yi = a[i]*(pts - x[i])**3 + b[i]*(pts - x[i])**2 + c[i]*(pts - x[i]) + d[i]
    y.extend(yi)

    return y

def cardinal_cubic_basis(x, knots):
    x = np.asarray(x)
    knots = np.asarray(knots)
    n_knots = len(knots)
    n_points = len(x)
    
    # Initialize basis matrix
    basis = np.zeros((n_points, n_knots))
    
    def M1(t):
        return (1 - t)**3 / 6

    def M2(t):
        return (3*t**3 - 6*t**2 + 4) / 6

    def M3(t):
        return (-3*t**3 + 3*t**2 + 3*t + 1) / 6

    def M4(t):
        return t**3 / 6
    
    # Construct basis functions
    for i in range(n_points):
        for j in range(n_knots-1):
            # Find which knot interval contains the current x value
            if knots[j] <= x[i] <= knots[j+1]:
                # Compute normalized position within interval
                t = (x[i] - knots[j]) / (knots[j+1] - knots[j])
                
                # Add contributions to relevant basis functions
                if j >= 1:
                    basis[i, j-1] += M1(t)
                basis[i, j] += M2(t)
                basis[i, j+1] += M3(t)
                if j < n_knots-2:
                    basis[i, j+2] += M4(t)
    
    return basis

def second_derivative_penalty_matrix(knots):
    knots = np.asarray(knots)
    n_knots = len(knots)
    P = np.zeros((n_knots, n_knots))
    
    def M1_dd(t):
        return 1-np.ones_like(t)

    def M2_dd(t):
        return 3*t - 2

    def M3_dd(t):
        return -3*t + 1

    def M4_dd(t):
        return np.ones_like(t)
    
    # List of second derivative functions
    M_dd = [M1_dd, M2_dd, M3_dd, M4_dd]
    
    # Construct penalty matrix by evaluating integrals over each knot interval
    for interval in range(n_knots - 1):
        h = knots[interval + 1] - knots[interval]
        
        # For each pair of basis functions that overlap on this interval
        for i in range(max(0, interval-2), min(n_knots, interval+3)):
            for j in range(max(0, interval-2), min(n_knots, interval+3)):
                # Determine which parts of the cardinal basis functions we're using
                i_idx = i - interval + 1
                j_idx = j - interval + 1
                
                if 0 <= i_idx <= 3 and 0 <= j_idx <= 3:
                    # Define the integrand as the product of second derivatives
                    def integrand(t):
                        return M_dd[i_idx](t) * M_dd[j_idx](t)
                    
                    # Scale the integral by the knot spacing
                    scaling = 1.0 / (h**3)
                    
                    # Compute the integral using scipy's quad
                    result, _ = integrate.quad(integrand, 0, 1)
                    P[i, j] += result * scaling

    return P

def driver():
    f = lambda x: np.cos(x)
    x0 = np.linspace(-np.pi, np.pi, 1000)
    x = np.arange(-np.pi,np.pi,0.5)

    (a,b,c,d) = get_cubic_spline_coefs(f, x)
    y = eval_cubic_splines(x0, x, a, b, c, d)

    plt.plot(x0, f(x0), label="function")
    plt.plot(x0, y, label="me")
    plt.legend()
    plt.show()

#if __name__ == "__main__":
#    driver()