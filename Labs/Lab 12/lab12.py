import numpy as np

def comp_trap(a, b, f, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

def comp_simpsons(a, b, f, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])