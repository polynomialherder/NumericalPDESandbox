import math

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm

def fft_solve(solver, n):
    solver.rows = n
    # Since the boundary conditions in the setup above are periodic, F(x) = f(x)
    # for all x in the domain (there are no "correction" terms at the boundaries)
    transformed = fft(solver.F)
    shifted = fftshift(transformed)
    k = -(solver.rows // 2)
    for idx, _ in enumerate(shifted):
        if not k:
            k += 1
            continue
        shifted[idx] += (shifted[idx] * (solver.upper_bound - solver.lower_bound)**2)/(4*np.pi*k**2)
        #print(k, shifted[idx], idx)
        k += 1
    #print("---")
    return ifft(shifted)

def fft_solve2(solver, n):
    solver.rows = n
    # Since the boundary conditions in the setup above are periodic, F(x) = f(x)
    # for all x in the domain (there are no "correction" terms at the boundaries)
    transformed = fft(solver.F)
    shifted = fftshift(transformed)
    midpoint = solver.rows // 2
    k = 0
    for idx, element in enumerate(transformed):
        if not k:
            k += 1
            continue
        transformed[idx] += (element * (solver.upper_bound - solver.lower_bound)**2)/(4*np.pi*k**2)
        #print(k, transformed[idx], idx)
        if idx == midpoint:
            k = -midpoint
            continue
        k += 1
    #print("---")
    return ifft(transformed)


if __name__ == '__main__':
    f = lambda x: -4*np.sin(2*x)
    u = lambda x: np.sin(2*x)
    alpha = BoundaryCondition(BCType.PERIODIC)
    beta = BoundaryCondition(BCType.PERIODIC)
    solver = PoissonSolver(f, rows=10, alpha=alpha, beta=beta, lower_bound=0, upper_bound=2*math.pi, actual=u)
    #solver.plot_h_vs_error()
    test_solution = fft_solve(solver, 9)
    test_solution2 = solver.fft_solution

