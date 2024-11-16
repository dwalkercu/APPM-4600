import numpy as np

def eval_comp_trap(a, b, f, N):
    h = b - a / N
    xi = np.linspace(a, b, N+1)
    Ihat = (h/2)*(f(a) + np.sum(2*f(xi)) + f(b))
    return Ihat

def eval_comp_simpsons(a, b, f, N):
    h = b - a / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    Ihat = (h/3)*(y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
    return Ihat

def find_eta(a, b, f, tol):
    new_eta = a
    x = np.arange(a, b, tol)
    for eta in x:
        if f(eta) < f(new_eta):
            new_eta = eta
    return new_eta if new_eta != a else np.nan

def choose_n_by_trap_error(a, b, f, tol):
    eta = find_eta(a, b, f, tol)
    if eta == np.nan: return -1
    fpp = np.gradient(np.gradient(f, eta), eta)
    h = (tol*12/(-(b-a)*fpp))**(-3)
    return h / (b-a)

''' Question 1 '''
# (a)
a = -5
b = 5
f = lambda s: 1/(1+s**2)
print(eval_comp_trap(a, b, f, 50))
print(eval_comp_simpsons(a, b, f, 50))

# (b)
print(choose_n_by_trap_error(a, b, f, 1e-4))