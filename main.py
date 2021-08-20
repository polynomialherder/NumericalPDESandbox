import math

import numpy as np

from solver.poisson_2d import PoissonSolver2D


def trig_factory(n, m, L, H, periods=1, rows_x=10, rows_y=10):
    coef_L = ((L)/(n*2*math.pi))**2
    coef_H = ((H)/(m*2*math.pi))**2
    f = lambda x, y: -(coef_L + coef_H)*np.sin(n*2*math.pi*x/L)*np.cos(m*2*math.pi*y/H)
    u = lambda x, y: np.sin(n*2*math.pi*x/L)*np.cos(m*2*math.pi*y/H)
    period_x = L/n
    period_y = H/m
    return PoissonSolver2D(f, u, rows_x, rows_y, 0, period_x*periods, 0, period_y*periods)


if __name__ == '__main__':
    f = lambda x, y: (12*x**2-24*x+8)*(y**4-4*y**3+4*y**2)+(x**4-4*x**3+4*x**2)*(12*y**2-24*y+8)
    actual = lambda x, y: (x**4-4*x**3+4*x**2)*(y**4-4*y**3+4*y**2) - (256/225)

    # Initialize a 2D solver
    p = PoissonSolver2D(f, actual, 10, 10, 0, 1, 0, 2)

    # Inspect the complex solution (inverse fft2 of fft2(F)*fourier_coefficients prior to coercion to a real-valued array)
    complex_solution = p.complex_solution

    # Inspect the real solution
    real_solution = p.U

    f2 = lambda x, y: 2*np.cos(2*x) + 2*np.cos(2*y)
    actual2 = lambda x, y: (np.sin(x))**2 + (np.sin(y))**2-1
    p2 = PoissonSolver2D(f2, actual2, 10, 10, 0, 2*math.pi, 0, 2*math.pi)

    p3 = trig_factory(3, 4, 5, 5)
    p4 = trig_factory(4, 2, 5, 5)
