import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def eval_monomial(xeval, coef, N, Neval):
    yeval = coef[0]*np.ones(Neval+1)
        
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def driver_1():
    N = 2
    Neval = 1000
    xeval = np.linspace(-1, 1, Neval+1)
    f = lambda x: 1/(1+100*x**2)
    x_j = lambda j: -1 + (j - 1)*(1/(N - 1))

    # create interpolation nodes
    xint = np.zeros(N+1)
    for j in range(N+1):
       xint[j] = x_j(j)

    yint = f(xint)

    # create the Vandermonde matrix
    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    Vinv = inv(V)
    coef = Vinv @ yint

    yeval = eval_monomial(xeval, coef, N, Neval)

if __name__ == "__main__":
    driver_1()