import numpy as np
import numpy.linalg as la
import math
import time

def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    4
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 6*x**3 + 2*np.sin(x)
    #y = f(x)
    #w = g(x)

    # make y,w orthogonal by utilizing unit vectors
    y = np.zeros(100)
    y[0] = 1
    w = np.zeros(100)
    w[1] = 1

    # initialize the matrix and vector for mat-vec multiplication
    matrix = [[1,2],[2,4]]
    z = [1,1]

    time_start = time.process_time()
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    mv = matVecMul(matrix, z, 2)
    time_without_np = time.process_time() - time_start

    time_start = time.process_time()
    dp_np = np.dot(y,w)
    mv_np = np.matmul(matrix, z)
    time_with_np = time.process_time() - time_start

    # print the output
    print('the dot product is : ', dp)
    print('the matrix-vector product is : ', mv)
    print('time it took: ', time_without_np)

    print('numpy operations:')
    print(f'dot product: {dp_np}')
    print(f'mat-vec multiplication: {mv_np}')
    print(f'time it took: {time_with_np}')

    print('I\'m guessing np is faster. could\'nt figure out time module')

    return

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0
    for j in range(n):
        dp = dp + x[j]*y[j]

    return dp

def matVecMul(x,y,n):
    # Computes the matrix-vector product of the n x n matrix and n x 1 vector
    mv = [0] * n
    for j in range(n):
        for x_i in range(n):
            # this adds the next term for the jth row in the resultant vector
            mv[j] += x[j][x_i] * y[x_i]
    
    return mv

driver()