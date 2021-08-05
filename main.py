import math

import numpy as np

from numpy.fft import fft2, fftshift, ifft2

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm

def fft_kinv_squared(k):
        midpoint = k // 2
        # kinv_squared is a generator with k-values in the right places per Python's fft transformation implementation
        # For example, given rows = 9, kinv_squared will return a generator containing the
        # following values:
        #   [0, (1/1)**2, (1/2)**2, (1/3)**2, (1/4)**2, (-1/4)**2, (-1/3)**2, (-1/2)**2, (-1/1)**2]
        kinv_squared = (0 if not i else (1/i)**2 if i <= midpoint else 1/(k-i)**2 for i in range(k))
        return np.fromiter(kinv_squared, dtype=float, count=k)



if __name__ == '__main__':
    # This "f" is just a placeholder until I come up with a valid test case
    f = lambda x, y: -4*np.sin(2*x*y)
    u = lambda x: np.sin(2*x)

    # Assume a square domain, so upper_x = upper_y
    # and lower_x = lower_y
    upper_bound = 2*math.pi
    lower_bound = 0
    rows = 10
    h = (upper_bound - lower_bound)/rows
    # Create a discretized domain corresponding to the square domain
    x_ = np.linspace(lower_bound + h/2, upper_bound - h/2, rows, endpoint=True)
    y_ = np.linspace(lower_bound + h/2, upper_bound - h/2, rows, endpoint=True)
    x, y = np.meshgrid(x_, y_)
    F = f(x,y)
    transformed = fft2(F)
    kinv_squared = fft_kinv_squared(rows)
    fourier_coefficients = ((upper_bound - lower_bound)**2)/(4*np.pi**2)
    U = np.real(ifft2(transformed*fourier_coefficients))
