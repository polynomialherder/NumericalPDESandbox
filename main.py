import math

import numpy as np

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm


if __name__ == '__main__':
    f = lambda x: -4*np.sin(2*x)
    u = lambda x: np.sin(2*x)
    alpha = BoundaryCondition(BCType.PERIODIC, 0)
    beta = BoundaryCondition(BCType.PERIODIC, 0)
    solver = PoissonSolver(f, rows=10, alpha=alpha, beta=beta, lower_bound=0, upper_bound=2*math.pi, actual=u)
    solver.plot_h_vs_error()

