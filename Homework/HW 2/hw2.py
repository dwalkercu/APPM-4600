import math
import random
import numpy as np
import matplotlib.pyplot as plt

### Q3
def f3(x):
    y = math.exp(x)
    return y-1

def poly_f3(x):
    return x + (1/2)*x**2 + (1/6)*x**3

print(f3(9.999999995000000*10**-10))
print(poly_f3(9.999999995000000*10**-10))


### Q4.a
t = np.arange(0, np.pi, np.pi / 30, dtype=float)
t = np.append(t, t[len(t)-1] + np.pi/30) # one more iteration
y = np.cos(t)

def sum_fn(y, t):
    sum = 0
    for i in range(0, len(t)-1):
        sum += t[i] * y[i]
    return sum

print(f"the sum is: {sum_fn(y, t)}")


### Q4.b1
R = 1.2
dr = 0.1
f = 15
p = 0

def x_fn(t):
    return R * (1 + dr * math.sin(f * t + p)) * math.cos(t)

def y_fn(t):
    return R * (1 + dr * math.sin(f * t + p)) * math.sin(t)

theta = np.linspace(0, 2*np.pi, 100)
xs = [x_fn(t) for t in theta]
ys = [y_fn(t) for t in theta]
plt.plot(xs, ys)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Wavy Circles')
plt.show()

### Q4.b2
plt.xlabel('x')
plt.ylabel('y')
plt.title('Wavy Circles: 10 Curves')

for i in range(0, 10):
    R = i
    dR = 0.05
    f = 2+i
    p = random.uniform(0, 2)

    xs = [x_fn(t) for t in theta]
    ys = [y_fn(t) for t in theta]

    plt.plot(xs, ys)

plt.show()