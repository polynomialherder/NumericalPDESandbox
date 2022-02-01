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
    a = 3
    b = 4
    u_actual = lambda x, y: L*np.sin(a*pi*x/L)*np.cos(b*pi*y/H)
    ux = lambda x, y: a*pi*np.cos(a*pi*x/L)*np.cos(b*pi*y/H)
    uy = lambda x, y: -(b*L*pi*np.sin(a*pi*x/L)*np.sin(b*pi*y/H))/H
    uxx = lambda x, y: -(a**2 * pi**2 * np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/L
    uyy = lambda x, y: -(b**2 * L * pi**2* np.cos(b*pi*y/H)*np.sin(a*pi*x/L))/H**2
    v_actual = lambda x, y: -H*np.cos(a*pi*x/L)*np.sin(b*pi*y/H)
    vy = lambda x, y: -2*N*pi*np.cos((2*N*pi*x)/L)*np.cos((2*N*pi*y)/H)
    vyy = lambda x, y: (b**2 * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/H
    vxx = lambda x, y: (a**2 * H * pi**2 * np.cos(a*pi*x/L)*np.sin(b*pi*y/H))/L**2
    f = lambda x, y: uxx(x, y) + uyy(x, y) - eta(L)(x)
    g = lambda x, y: vxx(x, y) + vyy(x, y) - eta(H)(y)
    return StokesSolver(
        f=f,
        g=g,
        u_actual=u_actual,
        v_actual=v_actual,
        rows_x=N,
        rows_y=N,
        x_lower=0,
        x_upper=L,
        y_lower=0,
        y_upper=H
    ), ux, vy


if __name__ == '__main__':
    s1, ux, vy = test_factory(1, 1, 400)
    plt.contourf(s1.x, s1.y, s1.u_actual)
    #plt.colorbar()
    #plt.contourf(s1.x, s1.y, s1.v)
    
