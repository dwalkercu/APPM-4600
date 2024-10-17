import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

def eval_line(x,x0,f_x0,x1,f_x1):
    return (1/(x1-x0))*(f_x0*(x1-x) + f_x1*(x-x0))

def eval_lin_spline(xint,xeval,Neval,a,b,f,Nint):
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval+1)
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
        '''temporarily store your info for creating a line in the interval of
        interest'''
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))
        for kk in range(n):
            #use your line evaluator to evaluate the spline at each location
            yloc[kk] = eval_line(xloc[kk], atmp, fa, btmp, fb) #Call your line evaluator with points (atmp,fa) and (btmp,fb)
            # Copy yloc into the final vector
            yeval[ind] = yloc
    return yeval

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)
    h[0] = xint[i]-xint[i-1]  
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

#  create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1))


#  Invert A    
    Ainv = inv(A)

# solver for M    
    M  = Ainv @ b
    
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = # find the C coefficients
       D[j] = # find the D coefficients
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
   
    yeval = 
    return yeval 
    
    
def eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)

def driver():
    N = 18
    Neval = 1000
    h = 2/(N-1)
    f = lambda x: 1 / (1 + 100*x**2)
    x_j = lambda j: -1 + (j - 1)*h

    xint = np.zeros(N+1)

    for j in range(N+1):
        xint[j] = x_j(j)

    yint = f(xint)

    xeval = np.linspace(-1, 1, Neval+1)
    yexact = f(xeval)

    # linear spline
    yeval = eval_lin_spline(xint, xeval, Neval, -1, 1, f, N)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Spline Interpolation")

    plt.plot(xeval, yeval, 'r-')
    plt.plot(xeval, yexact)
    plt.legend(['Interpolation', 'Exact'])

    plt.figure()
    plt.title("Linear Spline Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.semilogy(xeval, abs(yeval - yexact))

    plt.show()

    # cubic spline
    (M,C,D) = create_natural_spline(yint, xint, N)
    yeval = eval_cubic_spline(xeval, Neval, xint, N, M, C, D)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Cubic Spline Interpolation")

    plt.plot(xeval, yeval, 'ro-')
    plt.plot(xeval, yexact, 'bs--')
    plt.legend(['Interpolation', 'Exact'])

    plt.figure()
    plt.title("Cubic Spline Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.semilogy(xeval, abs(yeval - yexact))

    plt.show()


if __name__ == "__main__":
    driver()