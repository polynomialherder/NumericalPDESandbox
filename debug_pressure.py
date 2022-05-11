import math

from math import pi

import numpy as np

from solver.stokes import StokesSolver

import matplotlib.pyplot as plt

def eta(L, b=-2):
    """ This constructs a linear function in one variable that
        will serve as one of the components of the pressure gradient
    """
    print(f"{-2*b}/{L} x + {b}")
    return lambda x: -2*b*x/L + b


def test_factory(L, H, N):
    a = 2
    b = 2
    p_actual = lambda x, y: -0*x/L - 0
    p_x = lambda x, y: -2/L
    p_y = lambda x, y: 0
    u_actual = lambda x, y: L*np.sin(a*pi*x/L)*np.cos(b*pi*y/H)
    ux = lambda x, y: a*pi*np.cos(a*pi*x/L)*np.cos(b*pi*y/H)
    uy = lambda x, y: -(b*L*pi*np.sin(a*pi*x/L)*np.sin(b*pi*y/H))/H
    uxx = lambda x, y: -(a**2 * pi**2 * np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/L
    uyy = lambda x, y: -(b**2 * L * pi**2* np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/H**2
    v_actual = lambda x, y: -H*np.cos(a*pi*x/L)*np.sin(b*pi*y/H)
    vy = lambda x, y: -2*N*pi*np.cos((2*N*pi*x)/L)*np.cos((2*N*pi*y)/H)
    vyy = lambda x, y: (b**2 * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/H
    vxx = lambda x, y: (a**2 * H * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/L**2
    f = lambda x, y: uxx(x, y) + uyy(x, y) - p_x(x, y)
    g = lambda x, y: vxx(x, y) + vyy(x, y) - p_y(x, y)
    f = lambda x, y: x - 0.5
    g = lambda x, y: y - 0.5
    X = np.linspace(0, 1, N)
    Y = np.linspace(0, 1, N)
    xv, yv = np.meshgrid(X, Y)
    F = f(xv, yv)
    G = g(xv, yv)
    U = u_actual(xv, yv)
    V = v_actual(xv, yv)
    P = p_actual(xv, yv)
    return StokesSolver(xv, yv, F=F, G=G,
                        u_actual=U, v_actual=V,
                        p_actual=P)

if __name__ == '__main__':
    f = lambda x, y: (x - 0.5)
    g = lambda x, y: (y - 0.5)
    p = lambda x, y: 0.5*((x - 0.5)**2 + (y - 0.5)**2) - 1/6

    X = np.linspace(0, 1, 1000)
    Y = np.linspace(0, 1, 1000)
    xv, yv = np.meshgrid(X, Y)
    F = f(xv, yv)
    G = g(xv, yv)
    U = np.zeros(xv.shape)
    V = np.zeros(yv.shape)
    P = p(xv, yv)
    s1 = StokesSolver(
        xv, yv, F=F, G=G, u_actual=U, v_actual=V, p_actual=P
    )
    cm = plt.contourf(s1.x, s1.y, s1.p)
    plt.colorbar()
    #plt.show()
    print(s1.error_p(2))
