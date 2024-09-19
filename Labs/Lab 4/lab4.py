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

def order_of_convergence(phat):
  if (type(phat) == np.ndarray and len(phat) < 3) or (type(phat) != np.ndarray):
    return -1

  alpha = 1
  p = phat[-1]
  pn1 = phat[-2]
  pn = phat[-3]
  nmax = 50

  count = 0
  lda = 0
  while(count < nmax):
    lda = (np.abs(pn1 - p)) / (np.abs(pn - p)**alpha)
    if lda < 1 and lda >= 0:
      return (alpha, lda)
    alpha += 1
    count += 1

  return -1

def test_func(x):
  return (10/(x+4))**(1/2)

# 2.2
its = fixedpt_its(test_func, 1.5, 10**-10, 50)
if type(its) != None:
  print(order_of_convergence(its))
else:
  print("fixed pt didn't converge")


# 3.2
