from solver.stokes import StokesSolver
from solver.ib_utils import spread_to_fluid


class Fluid:

    def __init__(self, xv, yv, mu=1, membrane=None):
        self.xv = xv
        self.yv = yv
        self.mu = mu
        self.membrane = membrane
        self.solver = StokesSolver(self.xv, self.yv, mu=mu)

    def register(self, membrane):
        self.membrane = membrane
        self.membrane.fluid = self

    def stokes_solve(self, fx, fy):
        self.solver.F = -fx
        self.solver.G = -fy
        return self.solver.u, self.solver.v, self.solver.p

    @property
    def shape(self):
        return self.xv.shape

    def spread(self, F):
        return spread_to_fluid(F, self, self.membrane)

    def __repr__(self):
        return f"<Fluid mu={self.mu}>"
