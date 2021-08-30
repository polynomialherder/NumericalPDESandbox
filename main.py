import math

import numpy as np

from solver.poisson_2d import PoissonSolver2D
from matplotlib import pyplot as plt

from sympy import Symbol, integrate, lambdify


def trig_factory(n, m, L, H, rows_x=10, rows_y=10):
    coef_L = ((n*2*math.pi)/(L))**2
    coef_H = ((m*2*math.pi)/(H))**2
    f = lambda x, y: -(coef_L + coef_H)*np.sin(n*2*math.pi*x/L)*np.cos(m*2*math.pi*y/H)
    u = lambda x, y: np.sin(n*2*math.pi*x/L)*np.cos(m*2*math.pi*y/H)
    period_x = L
    period_y = H
    return PoissonSolver2D(f, u, rows_x, rows_y, 0, period_x, 0, period_y)


def polynomial_factory(L, H, rows_x=10, rows_y=10):
    x = Symbol('x')
    y = Symbol('y')
    C = integrate((x**2*(x-L)**2)*(y**2*(y-H)**2), (x, 0, L), (y, 0, H))
    C = float(C)
    u = lambda x, y: x**2*(x-L)**2*y**2*(y-H)**2 - C/(H*L)
    f = lambda x, y: 2*(x**2*(H-y)**2*(L-x)**2 + 4*x**2*y*(y-H)*(L-x)**2 + 4*x*y**2*(H-y)**2*(x-L) + y**2*(H-y)**2*(L-x)**2 + x**2*y**2*(H-y)**2 + x**2*y**2*(L-x)**2)
    return PoissonSolver2D(f, u, rows_x, rows_y, 0, L, 0, H)


if __name__ == '__main__':
    p = polynomial_factory(10, 20, rows_x=10000, rows_y=10000)
