from bisection_example import bisection
from fixedpt_example import fixedpt
import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(x) - 2*x + 1

def f2(x):
    return (x-5)**9

def f2_exp(x):
    return x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3375000*x - 1562500

def f3(x):
    return x**3 - x - 4

def f4(x):
   return -16 + 6*x + 12/x

def f5(x):
   return (2/3)*x + 1/(x**2)

def f6(x):
   return 12/(1+x)

def f7(x):
   return x - 4*np.sin(2*x) - 3

def f7_it(x):
   return x + (-np.sin(2*x) + x/4 - 3/4)

def fixedpt_its(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    its = np.zeros(Nmax)
    its[0] = x0
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if count < len(its):
        its[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          its = np.trim_zeros(its, 'b')
          if xstar == 0:
            its[len(its)-1] = 0
          return its
       x0 = x1

    xstar = x1
    ier = 1
    return None

def order_of_convergence(x, xstar):
  diff1 = np.abs(x[1::]-xstar)
  diff2 = np.abs(x[0:-1]-xstar)
  fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
  return (fit[0],np.exp(fit[1])) # (alpha, lambda)

print("--- Q1 ---")
print(bisection(f1, -1 * np.pi / 2, np.pi / 2, 1e-8))

print("--- Q2 ---")
print(bisection(f2, 4.82, 5.2, 1e-4))
print(bisection(f2_exp, 4.82, 5.2, 1e-4))

print("--- Q3 ---")
print(bisection(f3, 1, 4, 1e-3))

print("--- Q4 ---")
its = fixedpt_its(f4, 1.8, 1e-4, 100)
print(its)
if type(its) != type(None):
    print(order_of_convergence(its, 2.))
else:
   print("ERROR -- doesn't converge")

its = fixedpt_its(f5, 1, 1e-4, 100)
if type(its) != type(None):
    print(order_of_convergence(its, 3**(1/3)))
else:
   print("ERROR")

its = fixedpt_its(f6, 2, 1e-4, 100)
if type(its) != type(None):
    print(order_of_convergence(its, 3))
else:
   print("ERROR")

print("--- Q5 ---")
x = np.linspace(-np.pi, 2*np.pi, 100)
plt.plot(x, f7(x))
plt.plot(x, np.zeros(100))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Zeros of f(x)")
plt.show()

print(fixedpt(f7_it, -0.80, 1e-11, 100))
print(fixedpt(f7_it, 2.2, 1e-11, 100))