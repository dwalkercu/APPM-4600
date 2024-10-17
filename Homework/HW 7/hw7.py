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
    N = 18
    Neval = 1000
    xeval = np.linspace(-1, 1, Neval+1)
    f = lambda x: 1/(1+100*x**2)
    x_j = lambda j: -1 + (j - 1)*(2/(N - 1))

    # create interpolation nodes
    xint = np.zeros(N+1)
    for j in range(N+1):
       xint[j] = x_j(j)

    yint = f(xint)

    # create the Vandermonde matrix
    V = np.zeros((N+1,N+1))
    
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    Vinv = inv(V)
    coef = Vinv @ yint

    # create interpolation
    yeval = eval_monomial(xeval, coef, N, Neval)
    yexact = f(xeval)

    plt.title("Monomial Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xeval, yeval, 'r-')
    plt.plot(xeval, yexact, 'b-')
    plt.legend(["Interpolation", "Exact"])
    plt.show()

if __name__ == "__main__":
    driver_1()