import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

''' Problem 1 / Problem 2 '''

N = 10
Neval = 1000
f = lambda x: 1 / (1 + x**2)
fp = lambda x: -2*x / (1 + x**2)**2
x_cc = lambda x: -5*np.cos(np.pi * (2*x - 1) / (2*N))

# equispaced nodes
xint = np.linspace(-5, 5, N+1)

# Chebychev nodes
'''xint = np.zeros(N+1)

for i in range(N+1):
    xint[i] = x_cc(i)

xint[0] = -5 # cos is even so x = 0 and x = 1 produces the same value from x_cc'''

yint = f(xint)
yintp = fp(xint)

xeval = np.linspace(-5, 5, Neval+1)
yeval = np.zeros(Neval+1)
yexact = f(xeval)

# Lagrange Interpolation
def eval_lagrange(x, xint, yint, N):
    yeval = 0
    for i in range(N):
        L = 1
        for j in range(N):
            if i != j:
                L *= (x - xint[j]) / (xint[i] - xint[j])
        yeval += yint[i] * L
    return yeval

for kk in range(Neval+1):
    yeval[kk] = eval_lagrange(xeval[kk], xint, yint, N+1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange Interpolation")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Lagrange Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

# Hermite interpolation
def eval_hermite(x, xint, yint, yintp, N):
    yeval = 0
    for i in range(N):
        L = eval_lagrange(x, xint, yint, N)
        n = eval_lagrange(xint[i], xint, yint, N)
        H = (1 - 2*n*(x - xint[i])) * L**2
        K = (x - xint[i]) * L**2
        yeval += yint[i] * H + yintp[i] * K
    return yeval


yeval = np.zeros(Neval+1)

for kk in range(Neval+1):
    yeval[kk] = eval_hermite(xeval[kk], xint, yint, yintp, N+1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Hermite Interpolation")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Hermite Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

# Natural cubic spline interpolation
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)
    h[0] = xint[1]-xint[0]  
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

#  create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1)) # coefficient matrix for the second derivatives (nxn)

    for i in range(0,N):
        A[i,i] = (h[i]+h[i+1])/3
        if i > 0:
            A[i,i-1] = h[i]/6
        if i < N:
            A[i,i+1] = h[i+1]/6

    # fill out the N'th row
    A[N,N] = (h[N-1]+h[N])/3
    A[N,N-1] = h[N-1]/6

    # I don't know why this value isn't being filled by the for loop
    A[N-1,N] = h[N-1]/6

    # fix some more things
    A[0,0] = 1
    A[N,N] = 1
    A[0,1] = 0
    A[N,N-1] = 0

#  Invert A    
    Ainv = inv(A)

# solver for M   
    M  = Ainv @ b # vector containing second derivatives at every node (nx1)

    # enforce natural spline conditions
    M[0] = 0
    M[N] = 0
    
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = (yint[j]/h[j]) - (h[j]/6)*M[j] # find the C coefficients
       D[j] = (yint[j+1]/h[j]) - (h[j]/6)*M[j+1] # find the D coefficients
    return(M,C,D)
       

def eval_local_spline(xeval,xi,xip,yi,yip,Mi,Mip,Ci,Di):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    ai = Mi/6*hi
    bi = Mip/6*hi

    yeval = ai*(xip-xeval)**3 + bi*(xeval-xi)**3 + Ci*(xip-xeval) + Di*(xeval-xi)
    return yeval 
    
    
def eval_cubic_spline(xeval,Neval,xint,yint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,yint[j],yint[j+1],M[j],M[j+1],C[j],D[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)

M,C,D = create_natural_spline(yint,xint,N)
yeval = eval_cubic_spline(xeval,Neval,xint,yint,N,M,C,D)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Natural Cubic Spline Interpolation")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Natural Cubic Spline Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

# clamped cubic spline interpolation
def create_clamped_spline(yint,xint,N,yintp):
    #    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)
    h[0] = xint[1]-xint[0]  
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

#  create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1)) # coefficient matrix for the second derivatives (nxn)

    for i in range(0,N):
        A[i,i] = (h[i]+h[i+1])/3
        if i > 0:
            A[i,i-1] = h[i]/6
        if i < N:
            A[i,i+1] = h[i+1]/6

    # fill out the N'th row
    A[N,N] = (h[N-1]+h[N])/3
    A[N,N-1] = h[N-1]/6

    # I don't know why this value isn't being filled by the for loop
    A[N-1,N] = h[N-1]/6

    # fix some more things
    A[0,0] = 1
    A[N,N] = 1
    A[0,1] = 0
    A[N,N-1] = 0

#  Invert A    
    Ainv = inv(A)

# solver for M   
    M  = Ainv @ b # vector containing second derivatives at every node (nx1)

    # enforce clamped spline conditions
    M[0] = yintp[0]
    M[N] = yintp[N]
    
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = (yint[j]/h[j]) - (h[j]/6)*M[j] # find the C coefficients
       D[j] = (yint[j+1]/h[j]) - (h[j]/6)*M[j+1] # find the D coefficients
    return(M,C,D)

M,C,D = create_clamped_spline(yint,xint,N,yintp)
yeval = eval_cubic_spline(xeval,Neval,xint,yint,N,M,C,D)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Clamped Cubic Spline Interpolation")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Clamped Cubic Spline Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

''' Problem 3 '''
N = 20
f = lambda x: np.sin(10*x)
fp = lambda x: 10*np.cos(10*x)

def create_periodic_spline(yint,xint,N,yintp):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)
    h[0] = xint[1]-xint[0]  
    for i in range(1,N+1):
       h[i] = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1]

    #h[N] = xint[N+1] - xint[N]

#  create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1)) # coefficient matrix for the second derivatives (nxn)

    for i in range(0,N):
        A[i,i] = 2*(h[i]+h[i+1])
        if i > 0:
            A[i,i-1] = h[i]
        if i < N:
            A[i,i+1] = h[i+1]

    # fill out the N'th row
    A[N,N] = 2*(h[N-1]+h[N])
    A[N,N-1] = h[N-1]
    A[N-1,N] = h[N-1]

    # set the periodic conditions
    A[0,0] = 2*(h[0]+h[-1])
    A[0,-1] = h[-1]
    A[-1,0] = h[-1]
    A[-1,-1] = 2*(h[-1]+h[-2])
    b[0] = (3 / h[0]) * (yint[1] - yint[0]) - (3 / h[-1]) * (yint[0] - yint[-1])
    b[-1] = (3 / h[-1]) * (yint[0] - yint[-1]) - (3 / h[-2]) * (yint[-1] - yint[-2])

    print(A)

#  Invert A    
    Ainv = inv(A)

# solver for M   
    M  = Ainv @ b # vector containing second derivatives at every node (nx1)
    
#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = (yint[j]/h[j]) - (h[j]/6)*M[j] # find the C coefficients
       D[j] = (yint[j+1]/h[j]) - (h[j]/6)*M[j+1] # find the D coefficients
    return(M,C,D)

xint = np.linspace(0, 2*np.pi, (N+1)+1)
xeval = np.linspace(0, 2*np.pi, Neval+1)
yint = f(xint)
yintp = fp(xint)
yexact = f(xeval)

M,C,D = create_periodic_spline(yint,xint,N,yintp)
yeval = eval_cubic_spline(xeval,Neval,xint,yint,N,M,C,D)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation of a Periodic Function")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Periodic Interpolation Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()