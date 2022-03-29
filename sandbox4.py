from dataclasses import dataclass

import numpy as np

from scipy.linalg import norm

from solver.stokes import StokesSolver


class SimulationStep:

    def __init__(self, t, membrane, fluid, solver):
        self.membrane = membrane
        self.fluid = fluid
        self.solver = solver
        self.t = t


class Membrane:

    def __init__(self, X, Y, k, p=2):
        self.X = X
        self.Y = Y
        self.k = k
        self.p = p


    def difference_minus(self, vec):
        shifted = np.roll(vec, -1)
        return shifted - vec

    def difference_plus(self, vec):
        shifted = np.roll(vec, 1)
        return shifted - vec

    @property
    def difference_minus_x(self):
        return self.difference_minus(self.X)

    @property
    def difference_plus_x(self):
        return self.difference_plus(self.X)

    @property
    def norm_minus_x(self):
        return norm(self.difference_minus_x, self.p)

    @property
    def norm_plus_x(self):
        return norm(self.difference_minus_x, self.p)

    @property
    def tau_minus_x(self):
        return self.difference_minus_x/self.norm_minus_x

    @property
    def tau_plus_x(self):
        return self.difference_plus_x / self.norm_plus_x

    @property
    def difference_minus_y(self):
        return self.difference_minus(self.Y)

    @property
    def difference_plus_y(self):
        return self.difference_plus(self.Y)

    @property
    def norm_minus_y(self):
        return norm(self.difference_minus_y, self.p)

    @property
    def norm_plus_y(self):
        return norm(self.difference_plus_y, self.p)

    @property
    def tau_minus_y(self):
        return self.difference_minus_y / self.norm_minus_y

    @property
    def tau_plus_y(self):
        return self.difference_plus_y / self.norm_plus_y

    @property
    def tau_x(self):
        return self.tau_minus_x + self.tau_plus_x

    @property
    def tau_y(self):
        return self.tau_minus_y + self.tau_plus_y

    @property
    def dS_minus(self):
        return np.sqrt(self.difference_minus_x**2 + self.difference_minus_y**2)

    @property
    def dS_plus(self):
        return np.sqrt(self.difference_plus_x**2 + self.difference_plus_y**2)

    @property
    def dS(self):
        return (self.arc_length_minus + self.arc_length_plus)/2

    @property
    def Fx(self):
        return self.k*(self.tau_plus_x - self.tau_minus_x)/self.dS

    @property
    def Fy(self):
        return self.k*(self.tau_plus_y - self.tau_minus_y)/self.dS


if __name__ == '__main__':
    N = 2001

    theta = np.linspace(0, 2 * np.pi, N)
    theta = theta[0:-1]
    X = np.cos(theta) / 3 + 1 / 2
    Y = np.sin(theta) / 3 + 1 / 2

    kx = np.full(shape=N-1, fill_value=-0.8, dtype=np.float64)
    ky = np.full(shape=N-1, fill_value=-0.8, dtype=np.float64)
    k = ForceConstant(kx, ky)

    m = Membrane(X, Y, k)
