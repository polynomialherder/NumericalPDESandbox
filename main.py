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
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        actual=u,
        alpha=0,
        beta=20
    )
    eqn.dense = True
    eqn.plot_h_vs_error()


    alpha_ = BoundaryCondition(BCType.DIRICHLET, 0)
    beta_ = BoundaryCondition(BCType.NEUMANN, 20)
    u_ = lambda x: (1/20)*x*(x ** 4 + 395)
    eqn_ = PoissonSolver(
        f,
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        actual=u_,
        alpha=alpha_,
        beta=beta_
    )
    eqn_.dense = True
    eqn_.plot_h_vs_error()

