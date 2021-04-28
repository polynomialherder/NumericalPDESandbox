import math

import numpy as np

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm


if __name__ == '__main__':
    # Function representing choice of f(x) (as in the steady-state PDE u''(x) = f(x))
    f = lambda x: x ** 3
    # Function representing an analytic solution for u of the equation u'' = f(x)
    u = lambda x: (1/20)*x*(x ** 4 + 399)
    eqn = PoissonSolver(
        f,
        rows=10,
        lower_bound=0,
        upper_bound=1,
        actual=u,
        alpha=0,
        beta=20
    )
    eqn.dense = True
    #eqn.plot_h_vs_error()


    alpha_ = BoundaryCondition(BCType.DIRICHLET, 0)
    beta_ = BoundaryCondition(BCType.NEUMANN, 20)
    u_ = lambda x: (1/20)*x*(x ** 4 + 395)
    eqn_ = PoissonSolver(
        f,
        rows=10,
        lower_bound=0,
        upper_bound=1,
        actual=u_,
        alpha=alpha_,
        beta=beta_
    )
    eqn_.dense = False
    #eqn_.plot_h_vs_error()


    alpha3 = BoundaryCondition(BCType.NEUMANN, 0)
    beta3 = BoundaryCondition(BCType.DIRICHLET, 1)
    u3 = lambda x: (1/20)*(x ** 5 + 19)
    eqn3 = PoissonSolver(
        f,
        rows=10,
        lower_bound=0,
        upper_bound=1,
        actual=u3,
        alpha=alpha3,
        beta=beta3
    )
    eqn3.dense = False
    #eqn3.plot_h_vs_error()

    f4 = lambda t: np.cos(t)
    actual4 = lambda t: 1 - np.cos(t)
    alpha4 = BoundaryCondition(BCType.NEUMANN, 0)
    beta4 = BoundaryCondition(BCType.NEUMANN, 20)
    eqn4 = PoissonSolver(
        f4,
        rows=10,
        lower_bound=0,
        upper_bound=2*math.pi,
        actual=actual4,
        alpha=alpha4,
        beta=beta4
    )
    eqn4.plot_h_vs_error()
