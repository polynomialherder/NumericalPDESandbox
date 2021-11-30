import math

from math import pi

import numpy as np

from solver.stokes import StokesSolver

def test_factory(L, H, N):
    f = lambda x, y: uxx(x, y) + uyy(x, y) - 2
    u_actual = lambda x, y: L*np.sin(N*2*pi*x/L)*np.cos(N*2*pi*y/H)
    uxx = lambda x, y: -(4*(N**2)*pi**2)*np.cos(2*pi*N*y/H)*np.sin(2*pi*N*x/L)/L
    uyy = lambda x, y: -(4*(N**2)*pi**2)*L*np.cos(2*pi*N*y/H)*np.sin(2*pi*N*x/L)/H**2
    g = lambda x, y: vxx(x, y) + vyy(x, y) - 1
    v_actual = lambda x, y: -H*np.cos(N*2*pi*x/L)*np.sin(N*2*pi*y/H)
    vyy = lambda x, y: 4*(pi**2)*(N**2)*np.sin(2*pi*N*y/H)*np.cos(2*pi*N*x/L)/H
    vxx = lambda x, y: 4*(pi**2)*H*(N**2)*np.sin(2*pi*N*y/H)*np.cos(2*pi*N*x/L)/L**2
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
    )


if __name__ == '__main__':
    s1 = test_factory(10, 5, 100)
