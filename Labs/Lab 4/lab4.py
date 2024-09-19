import numpy as np

def fixedpt_its(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    its = np.zeros((Nmax))
    its[0] = x0
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       its[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          its = np.trim_zeros(its, 'b')
          if xstar == 0:
            its[its.len()-1] = 0
          return its
       x0 = x1

    xstar = x1
    ier = 1
    return ier

def test_func(x):
    return x - (x**5-7)/12

print(fixedpt_its(test_func, 1, 0.1, 50))