import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def eval_monomial(xeval, coef, N, Neval):
    yeval = coef[0]*np.ones(Neval+1)
        
    for j in range(1,N+1):
      for i in range(Neval+1):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def eval_barycentric(xeval, xint, yint):
    phi_n = 1

    # evaluate phi_n
    for i in range(len(xint)):
        phi_n *= (xeval - xint[i])
    
    # evaluate sum
    sum = 0
    for j in range(len(xint)):
        w_j = 1
        for i in range(len(xint)):
            if i != j:
                w_j *= (xint[j] - xint[i])
        w_j = 1/w_j

        sum += (w_j/(xeval - xint[j]))*yint[j]
    
    return phi_n*sum

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

def driver_2():
    N = 5
    Neval = 1000
    xeval = np.linspace(-1, 1, Neval+1)
    f = lambda x: 1/(1+100*x**2)
    #x_j = lambda j: -1 + (j - 1)*(2/(N - 1))
    x_j = lambda j: np.cos((2*j - 1)*np.pi/(2*N))

    # create interpolation nodes
    xint = np.zeros(N+1)
    for j in range(N+1):
       xint[j] = x_j(j)

    xint[0] = 1 # to prevent divide by 0 in barycentric formula
    yint = f(xint)

    # create interpolation
    yeval = eval_barycentric(xeval, xint, yint)
    yexact = f(xeval)

    plt.title("Barycentric Lagrange Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xeval, yeval, 'r-')
    plt.plot(xeval, yexact, 'b-')
    plt.legend(["Interpolation", "Exact"])
    plt.show()

    plt.title("Barycentric Lagrange Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.semilogy(xeval, abs(yeval - yexact))
    plt.show()

if __name__ == "__main__":
    #driver_1()
    driver_2()