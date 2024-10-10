import numpy as np
import matplotlib.pyplot as plt
import monomial_interp as mi

''' 3.1 '''

# monomial expansion

N = 10
Neval = 1000
h = 2/(N-1)
f = lambda x: 1 / (1 + 100*x**2)
x_j = lambda j: -1 + (j - 1)*h

xint = np.zeros(N+1)

for j in range(N):
    xint[j] = x_j(j)

yint = f(xint)

V = mi.Vandermonde(xint, N)
V_inv = np.linalg.inv(V)

coef = V_inv @ yint

xeval = np.linspace(-1, 1, Neval+1)
yeval = mi.eval_monomial(xeval, coef, N, Neval)

yexact = f(xeval)
err = np.linalg.norm(yexact - yeval)

# plot approximations & abs err
plt.xlabel("x")
plt.ylabel("y")
plt.title("Monomial Expansion Interpolation")

plt.plot(xeval, yeval)
plt.plot(xeval, yexact)
plt.plot(xeval, np.full(Neval+1, fill_value=err))
plt.show()