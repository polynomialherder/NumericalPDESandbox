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
    eqn_.dense = True
    #eqn_.plot_h_vs_error()

    f3 = lambda t: -(2*np.sin(t) + t*np.cos(t))
    actual3 = lambda t: t * np.cos(t)
    eqn3 = PoissonSolver(
        f3,
        rows=10,
        lower_bound=0,
        upper_bound=2*math.pi, #2*math.pi,
        actual=actual3,
        alpha=0,
        beta=2*math.pi
    )
    eqn3.dense = True
    #eqn3.plot_h_vs_error()

    f4 = lambda t: np.cos(t)
    actual4 = lambda t: 1 - np.cos(t)
    eqn4 = PoissonSolver(
        f4,
        rows=10,
        lower_bound=0,
        upper_bound=2*math.pi,
        actual=actual4,
        alpha=BoundaryCondition(
            BCType.PERIODIC,
            f4(2*math.pi)
        ),
        beta=BoundaryCondition(
            BCType.PERIODIC,
            f4(2*math.pi)
        )
    )
    eqn4.plot_h_vs_error()
