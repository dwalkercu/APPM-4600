import numpy as np

def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

def f1(x):
    return x**2 * (x - 1)

def f2(x):
    return (x-1)*(x-3)*(x-5)

def f3(x):
    return ((x-1)**2) * (x-3)

def f4(x):
    return np.sin(x)

def f5(x):
    return x * (1+(7-x**5)/x**2)**3

def f6(x):
    return x - (x**5-7)/x**2

def f7(x):
    return x - (x**5-7)/5*x**4

def f8(x):
    return x - (x**5-7)/12


### Q1

# (a)
print(f"Q1.a: {bisection(f1, 0.5, 2, 10**-5)}")

# (b)
print(f"Q1.b: {bisection(f1, -1, 0.5, 10**-5)}")

# (c)
print(f"Q1.c: {bisection(f1, -1, 2, 10**-5)}")



### Q2

# (a)
print(f"Q2.a: {bisection(f2, 0, 2.4, 10**-5)}")

# (b)
print(f"Q2.b: {bisection(f3, 0, 2, 10**-5)}")

# (c)
print(f"Q2.c1: {bisection(f4, 0, 0.1, 10**-5)}")
print(f"Q2.c2: {bisection(f4, 0.5, (3*np.pi)/4, 10**-5)}")

### Q3


# note: n_max = 3 was used for the diverging functions to get the code to run successfully
# (a)

print(f"Q3.a1: {fixedpt(f5, 7**(1/5), 10**-5, 50)}")
print(f"Q3.a2: {fixedpt(f5, 1, 10**-10, 3)}")

# (b)

print(f"Q3.b1: {fixedpt(f6, 7**(1/5), 10**-5, 50)}")
print(f"Q3.b2: {fixedpt(f6, 1, 10**-10, 3)}")

# (c)

print(f"Q3.c1: {fixedpt(f7, 7**(1/5), 10**-5, 50)}")
print(f"Q3.c2: {fixedpt(f7, 1, 10**-10, 3)}")

# (d)

print(f"Q3.d1: {fixedpt(f8, 7**(1/5), 10**-5, 50)}")
print(f"Q3.d2: {fixedpt(f8, 1, 10**-10, 50)}")