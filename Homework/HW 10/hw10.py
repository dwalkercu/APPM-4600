import numpy as np
import matplotlib.pyplot as plt

def approx_1(x):
    return (1 + x - (7/60)*x**3) / (1+(1/20)*x**2)

def approx_2(x):
    return x / (1 + (1/6)*x**2 + (1/36)*x**3 + (7/360)*x**4)

def approx_3(x):
    return (x - (7/60)*x**3) / (1 + (1/20)*x**2)

def mac(x):
    return x - x**3/6 + x**5/120

error1 = lambda x: np.abs(approx_1(x) - mac(x))
error2 = lambda x: np.abs(approx_2(x) - mac(x))
error3 = lambda x: np.abs(approx_3(x) - mac(x))

x = np.linspace(0, 5, 100)
plt.title('Approximation 1')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, approx_1(x), label='Approximation 1')
plt.plot(x, mac(x), label='Maclaurin')
plt.plot(x, error1(x), label='Error 1')
plt.legend()
plt.show()
plt.title('Approximation 2')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, approx_2(x), label='Approximation 2')
plt.plot(x, mac(x), label='Maclaurin')
plt.plot(x, error2(x), label='Error 2')
plt.legend()
plt.show()
plt.title('Approximation 3')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, approx_3(x), label='Approximation 3')
plt.plot(x, mac(x), label='Maclaurin')
plt.plot(x, error3(x), label='Error 3')
plt.legend()
plt.show()