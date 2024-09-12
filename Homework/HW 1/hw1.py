import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np

def q1():
    x = np.arange(1.920, 2.080, 0.001, dtype=float)
    y = (x-2)**9
    y_expanded = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

    # plot p via expanded form
    plt.plot(x, y_expanded)
    plt.title('p via Expanded Form')
    plt.xlabel('x')
    plt.ylabel('y')

    # plot p via compact form
    plt.plot(x, y)
    plt.title("p via Compact Form")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def q4():
    r1 = 28. + (1/2)*np.sqrt(3132)
    r2 = 28 - (1/2)*np.sqrt(3132)

    print(np.abs((55.982 - r1))/r1)
    print((0.018 - r2)/r2)
    print(1/r1)

def q5():

    ## part a

    x1 = 10**5
    dx1 = 50
    x2 = 2*10**5
    dx2 = 10**3

    x1t = x1 + dx1
    x2t = x2 + dx2

    y = x1t - x2t
    dy = dx1 - dx2
    
    #print(y)
    #print(np.abs(dy))
    #print(np.abs(dy)/np.abs(y))


    ## part b
    x1 = np.pi
    x2 = 10**6
    delta = [10**-i for i in range(0,17)]
    delta = np.asarray(delta[::-1])
    orig_exp1 = np.cos(x1 + delta) - np.cos(x1)
    orig_exp2 = np.cos(x2 + delta) - np.cos(x2)
    new_exp1 = -2*np.sin((2*x1 + delta)/2)*np.sin(delta / 2)
    new_exp2 = -2*np.sin((2*x2 + delta)/2)*np.sin(delta / 2)

    plt.xscale('log')
    plt.plot(delta, np.abs(orig_exp1 - new_exp1), label='x=pi')
    plt.plot(delta, np.abs(orig_exp2 - new_exp2), label='x=10^6')
    plt.legend(loc='upper left')
    plt.title("Abs Difference between Original and New Expression")
    plt.xlabel("delta")
    plt.ylabel("difference")
    plt.show()

    
    ## part c

    eta1 = [x1 for i in range(0, len(delta))]
    first_term1 = -1*delta*np.sin(x1)
    second_term1 = -1*((delta**2)/2)*np.cos(eta1)
    diff1 = np.abs(first_term1 + second_term1)

    eta2 = [x2 for i in range(0, len(delta))]
    first_term2 = -1*delta*np.sin(x2)
    second_term2 = -1*((delta**2)/2)*np.cos(eta2)
    diff2 = np.abs(first_term2 + second_term2)

    plt.xscale('log')
    plt.plot(delta, diff1, label='x=pi')
    plt.plot(delta, diff2, label='x=10^6')
    plt.show()

# q1()
# q4()
q5()