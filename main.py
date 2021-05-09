import math

import numpy as np

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm


if __name__ == '__main__':
    # Function representing choice of f(x) (as in the steady-state PDE u''(x) = f(x))
    f = lambda x: 2 * x
    # Function representing an analytic solution for u of the equation u''(x) = f(x)
    actual = lambda x: (1/3)*(-0.2497 + x**3)
    alpha = BoundaryCondition(BCType.NEUMANN, 0)
    beta = BoundaryCondition(BCType.NEUMANN, 1)
    solver = PoissonSolver(f, rows=10, alpha=alpha, beta=beta, lower_bound=0, upper_bound=1, actual=actual)
    solver.plot_h_vs_error()
