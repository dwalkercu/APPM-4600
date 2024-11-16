import numpy as np
from scipy.integrate import quad

def eval_comp_trap(a, b, f, N):
    h = (b - a) / N
    xi = np.linspace(a, b, N+1)
    Ihat = (h/2)*(f(a) + np.sum(2*f(xi)) + f(b))
    return Ihat

def eval_comp_simpsons(a, b, f, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    Ihat = (h/3)*(y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
    return Ihat

def find_eta(a, b, f, tol):
    new_eta = a
    x = np.arange(a, b, tol)
    for eta in x:
        if np.abs(f(eta)) > np.abs(f(new_eta)):
            new_eta = eta
    return new_eta

def choose_n_by_trap_error(a, b, f2p, tol):
    eta = find_eta(a, b, f2p, tol)
    if eta == np.nan: return -1
    h = (np.abs(tol*12/((b-a)*f2p(eta))))**(1/2)
    return int((b-a) / h)

def choose_n_by_simpsons_error(a, b, f4p, tol):
    eta = find_eta(a, b, f4p, tol)
    if eta == np.nan: return -1
    h = (np.abs(tol*180/((b-a)*f4p(eta))))**(1/4)
    return int((b-a) / h)

''' Question 1 '''

# (a)
a = -5
b = 5
tol = 1e-4
f = lambda s: 1/(1+s**2)
f2p = lambda s: 2*(3*s**2 - 1)/(1+s**2)**3
f4p = lambda s: 24*(5*s**4 - 10*s**2 + 1)/(1+s**2)**5
print(f"Eval Composite Trap: {eval_comp_trap(a, b, f, 50)}")
print(f"Eval Composite Simpsons: {eval_comp_simpsons(a, b, f, 50)}")

# (b)
N_trap = choose_n_by_trap_error(a, b, f, tol)
N_simps = choose_n_by_simpsons_error(a, b, f, tol)
print(f"N for trap ({tol}): {N_trap}")
print(f"N for simpsons ({tol}): {N_simps}")
print(f"Eval Composite Trap: {eval_comp_trap(a, b, f, N_trap)}")
print(f"Eval Composite Simpsons: {eval_comp_simpsons(a, b, f, N_simps)}")

# (c)
spquad1, err, info = quad(f, a, b, epsabs=tol, epsrel=tol, full_output=True)
print(f"SciPy Quad ({tol}): I={spquad1}, nevals={info.get('neval')}")
tol = 1e-6
spquad2, err, info = quad(f, a, b, epsabs=tol, epsrel=tol, full_output=True)
print(f"SciPy Quad ({tol}): I={spquad2}, nevals={info.get('neval')}")

''' Question 2 '''

