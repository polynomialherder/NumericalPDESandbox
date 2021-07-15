import math

import numpy as np

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm

if __name__ == '__main__':
    # Use cos for f and u
    # Make sure that domain is an integer number of periods
    # Different values of rows
    f = lambda x: -4*np.sin(2*x)
    u = lambda x: np.sin(2*x)
    alpha = BoundaryCondition(BCType.PERIODIC)
    beta = BoundaryCondition(BCType.PERIODIC)
    solver = PoissonSolver(
        f, rows=10, alpha=alpha, beta=beta, lower_bound=0, upper_bound=2*math.pi, actual=u
    )
    # Will automatically use fft_solution here, since our BCs are periodic
    test_solution2 = solver.solution

