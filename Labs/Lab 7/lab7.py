import numpy as np
import matplotlib.pyplot as plt
import monomial_interp as mi
import interp as ip

''' 3.1 '''

N = 18
Neval = 1000
h = 2/(N-1)
f = lambda x: 1 / (1 + 100*x**2)
x_j = lambda j: -1 + (j - 1)*h

xint = np.zeros(N+1)

for j in range(N+1):
    xint[j] = x_j(j)

yint = f(xint)

# monomial expansion

V = mi.Vandermonde(xint, N)
V_inv = np.linalg.inv(V)

coef = V_inv @ yint

xeval = np.linspace(-1, 1, Neval+1)
yeval = mi.eval_monomial(xeval, coef, N, Neval)

yexact = f(xeval)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Monomial Expansion Interpolation")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Monomial Expansion Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

# lagrange polynomials

yeval = np.zeros(Neval+1)

for kk in range(Neval+1):
    yeval[kk] = ip.eval_lagrange(xeval[kk], xint, yint, N)

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

# newton divided differences

yeval = np.zeros(Neval+1)

y = np.zeros( (N+1, N+1) )
     
for j in range(N+1):
   y[j][0]  = yint[j]

y = ip.dividedDiffTable(xint, y, N+1)

for kk in range(Neval+1):
    yeval[kk] = ip.evalDDpoly(xeval[kk], xint, y, N)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton Divided Differences")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Newton Divided Differences Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

''' 3.2 '''

x_j2 = lambda j: np.cos(((2*j - 1)*np.pi) / (2*N))

xint = np.zeros(N+1)

for j in range(N+1):
    xint[j] = x_j2(j)

xint[0] = 1 # to avoid division by zero in interp.py line 70

yint = f(xint)

xeval = np.linspace(-1, 1, Neval+1)
yexact = f(xeval)

# lagrange polynomials v2

yeval = np.zeros(Neval+1)

for kk in range(Neval+1):
    yeval[kk] = ip.eval_lagrange(xeval[kk], xint, yint, N)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange Interpolation V2")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Lagrange Interpolation V2 Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

# newton divided differences v2

yeval = np.zeros(Neval+1)

y = np.zeros( (N+1, N+1) )
     
for j in range(N+1):
   y[j][0]  = yint[j]

y = ip.dividedDiffTable(xint, y, N+1)

for kk in range(Neval+1):
    yeval[kk] = ip.evalDDpoly(xeval[kk], xint, y, N)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton Divided Differences V2")

plt.plot(xeval, yeval, 'r-')
plt.plot(xeval, yexact)
plt.legend(['Interpolation', 'Exact'])

plt.figure()
plt.title("Newton Divided Differences V2 Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.semilogy(xeval, abs(yeval - yexact))

plt.show()

