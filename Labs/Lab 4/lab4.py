import numpy as np

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

def fixedpt_steffensons(f, x0, tol, Nmax):
    its = np.zeros(Nmax)
    pn = x0
    count = 0

    while (count < Nmax):
      a = pn
      its[count] = a
      b = f(pn)
      c = f(b)

      if np.abs(a-b) < tol:
        its = np.trim_zeros(its, 'b')
        return its
      
      pn = a - ((b-a)**2) / (c - 2*b + a)
      count += 1

    return None

def order_of_convergence(x, xstar):
  diff1 = np.abs(x[1::]-xstar)
  diff2 = np.abs(x[0:-1]-xstar)
  fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
  return (fit[0],np.exp(fit[1])) # (alpha, lambda)

def delta_squared(seq, tol, nmax):
  count = 0
  p = seq[-1]
  its = np.zeros(nmax)

  while(count < nmax):
    pn = seq[count]
    pn1 = seq[count+1]
    pn2 = seq[count+2]

    phat = pn - (pn1 - pn)**2 / (pn2 - 2*pn1 + pn)
    its = np.append(its, phat)

    if(phat - p < tol):
      its = np.trim_zeros(its, 'b')
      if phat == 0:
        its[-1] = 0
      return its

    count += 1
  
  return None

def f1(x):
  return (10/(x+4))**(1/2)

# 2.2
its = fixedpt_its(f1, 1.5, 10**-10, 50)
if type(its) != None:
  print(order_of_convergence(its, 1.3652300134140976))
else:
  print("fixed pt didn't converge")


# 3.2
d2 = delta_squared(its, 10**-10, 50)
if type(d2) != None:
  print(order_of_convergence(d2, 1.3652300134140976))
else:
  print("error with delta_squared")

# 3.4
its_s = fixedpt_steffensons(f1, 1.5, 10**-10, 50)
if type(its_s) != None:
  print(order_of_convergence(its_s, 1.3652300134140976))
else:
  print("fixed pt steffenson's didn't converge")