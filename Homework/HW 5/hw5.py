import numpy as np
import newtonNONLinear2 as nN2

def order_of_convergence(x, xstar):
  diff1 = np.abs(x[1::]-xstar)
  diff2 = np.abs(x[0:-1]-xstar)
  fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
  return (fit[0],np.exp(fit[1])) # (alpha, lambda)

### Q1 ###
def it_scheme(f, g, x0, y0, tol, nmax):
    xy = [x0, y0]
    x_list = [x0]
    y_list = [y0]
    M = [[1/6, 1/18], [0, 1/6]]

    for i in range(nmax):
        xy = np.subtract(xy, np.matmul(M, [f(xy[0], xy[1]), g(xy[0], xy[1])]))
        x_list.append(xy[0])
        y_list.append(xy[1])

        if np.linalg.norm(np.abs(np.subtract(xy, [x_list[-2], y_list[-2]]))) < tol:
            return [xy, i, x_list, y_list]
    
    return [None, -1, None, None]

def f_1(x, y):
    return 3*x**2 - y**2

def g_1(x, y):
    return 3*x*y**2 - x**3 - 1

def evalF_Q1(x):
    return np.array([f_1(x[0], x[1]), g_1(x[0], x[1])])

def evalJ_Q1(x):
    return [[6*x[0], -2*x[1]], [3*x[1]**2, 3*x[1]**2 - 3*x[0]**2]]

print("### Q1 ###")
res = it_scheme(f_1, g_1, 1, 1, 1e-10, 100)
if(res[0][0] != None and res[0][1] != None):
    print("x = %.10f, y = %.10f, n = %i" % (res[0][0], res[0][1], res[1]))
else:
    print("No solution found")

res_newton = nN2.Newton([1, 1], 1e-10, 100, evalF_Q1, evalJ_Q1)
print("Newton's method: x = %.10f, y = %.10f, n = %i" % (res_newton[0][0], res_newton[0][1], res_newton[2]))

print("verification of solution: %.10f %.10f" % (f_1(res_newton[0][0], res_newton[0][1]), g_1(res_newton[0][0], res_newton[0][1])))


### Q2 ###

def f_2(x, y):
    return (1/np.sqrt(2))*np.sqrt(1 + (x**2 + y**2)**2) - (2/3)

def g_2(x, y):
    return (1/np.sqrt(2))*np.sqrt(1 + (x**2 - y**2)**2) - (2/3)

def fixed_pt_q2(x0, y0, tol, nmax):
    xy = [x0, y0]
    xy_bku = xy

    for i in range(nmax):
        xy = [f_2(xy[0], xy[1]), g_2(xy[0], xy[1])]

        if np.linalg.norm(np.abs(np.subtract(xy, xy_bku))) < tol:
            return [xy, i]
        
        xy_bku = xy
    
    return [None, -1]

print("\n### Q2 ###")

res1 = fixed_pt_q2(0, 0, 1e-10, 100)
res2 = fixed_pt_q2(0.5, 0.5, 1e-10, 100)
res3 = fixed_pt_q2(1.2, 1.2, 1e-10, 100)
res4 = fixed_pt_q2(0.33, 1.4, 1e-10, 100)
res5= fixed_pt_q2(1.5, -0.8, 1e-10, 100)

print(res1)
print(res2)
print(res3)
print(res4)
print(res5)


### Q3 ###
def srfc_newton(x0, y0, z0, tol, nmax):
    xyz = [x0, y0, z0]
    xyz_bku = xyz
    x_list = [x0]
    y_list = [y0]
    z_list = [z0]

    for i in range(nmax):
        x, y, z = xyz_bku
        d = (x**2 + 4*y**2 + 4*z**2 - 16) / (4*x**2 + 16*y**2 + 16*z**2)
        xyz[0] = x - d*(2*x)
        xyz[1] = y - d*(8*y)
        xyz[2] = z - d*(8*z)

        x_list.append(xyz[0])
        y_list.append(xyz[1])
        z_list.append(xyz[2])

        f = (xyz[0]**2 + 4*xyz[1]**2 + 4*xyz[2]**2 - 16)
        if f < tol:
            return [xyz, i, x_list, y_list, z_list]
        
        xyz_bku = xyz
    
    return [None, -1, None, None, None]

print("\n### Q3 ###")
res = srfc_newton(1, 1, 1, 1e-10, 100)
print("Surface Newton: ", (res[0], res[1]))