import math

import numpy as np

from solver.poisson_2d import PoissonSolver2D



if __name__ == '__main__':
    f = lambda x, y: (12*x**2-24*x+8)*(y**4-4*y**3+4*y**2)+(x**4-4*x**3+4*x**2)*(12*y**2-24*y+8)
    actual = lambda x, y, area: (x**4-4*x**3+4*x**2)*(y**4-4*y**3+4*y**2) - (256/225)/area

    # Initialize a 2D solver
    p = PoissonSolver2D(f, 10, 10, 0, 1, 0, 2, actual)

    # Inspect the complex solution (inverse fft2 of fft2(F)*fourier_coefficients prior to coercion to a real-valued array)
    complex_solution = p.complex_solution

    # Inspect the real solution
    real_solution = p.U

    f2 = lambda x, y: 2*np.cos(2*x) + 2*np.cos(2*y)
    actual2 = lambda x, y, area: (np.sin(x))**2 + (np.sin(y))**2-1
    p2 = PoissonSolver2D(f2, 10, 10, 0, 2*math.pi, 0, 2*math.pi, actual2)
