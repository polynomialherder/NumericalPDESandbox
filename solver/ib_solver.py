from dataclasses import dataclass
from math import floor
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm
from solver.ib_utils import spread_to_fluid, interp_to_membrane
from solver.stokes import StokesSolver


@dataclass
class SimulationStep:

    xv: np.array
    yv: np.array
    fx: np.array
    fy: np.array
    X: np.array
    Y: np.array
    t: float
    p: float

    def plot(self):
        plt.plot(self.X, self.Y, 'o')
        ax = plt.gca()
        ax.set_title(f"t={self.t}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid()
        plt.draw()
        plt.pause(0.1)
        plt.clf()


    def plot_pressure(self):
        cm = plt.pcolor(self.xv, self.yv, self.p)
        ax = plt.gca()
        ax.set_title(f"t={self.t}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.colorbar(cm)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


class Simulation:

    def __init__(self, fluid, membrane, dt, t=0, mu=1):
        self.fluid = fluid
        self.membrane = membrane
        self.dt = dt
        self.mu = mu
        self.t = t


    def calculate_forces(self):
        return self.membrane.Fx, self.membrane.Fy

    def spread_forces(self, Fx, Fy):
        return self.fluid.spread(Fx), self.fluid.spread(Fy)

    def stokes_solve(self, fx, fy):
        return self.fluid.stokes_solve(fx, fy)

    def calculate_velocities(self, u, v):
        return self.membrane.interp(u), self.membrane.interp(v)

    def update_membrane_positions(self, U, V):
        self.membrane.X += self.dt*U
        self.membrane.Y += self.dt*V

    def step(self):
        Fx, Fy = self.calculate_forces()
        fx, fy = self.spread_forces(Fx, Fy)
        u, v, p = self.stokes_solve(fx, fy)
        U, V = self.calculate_velocities(fx, fy)
        self.update_membrane_positions(U, V)
        self.t += self.dt
        return SimulationStep(
            self.fluid.xv, self.fluid.yv, fx, fy, self.membrane.X, self.membrane.Y, self.t, p
        )


class Fluid:

    def __init__(self, xv, yv, mu=1, membrane=None):
        self.xv = xv
        self.yv = yv
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


class Membrane:

    def __init__(self, X, Y, k, fluid=None, p=2):
        self.X = X
        self.Y = Y
        self.k = k
        self.p = p
        self.fluid = fluid


    def interp(self, f):
        return interp_to_membrane(f, self.fluid, self)


    def difference_minus(self, vec):
        shifted = np.roll(vec, 1)
        return vec - shifted

    def difference_plus(self, vec):
        shifted = np.roll(vec, -1)
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
        return self.difference_minus_x / self.dS_minus

    @property
    def tau_plus_x(self):
        return self.difference_plus_x / self.dS_plus

    @property
    def difference_minus_y(self):
        return self.difference_minus(self.Y)

    @property
    def difference_plus_y(self):
        return self.difference_plus(self.Y)

    @property
    def tau_minus_y(self):
        return self.difference_minus_y / self.dS_minus

    @property
    def tau_plus_y(self):
        return self.difference_plus_y / self.dS_plus

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
        return (self.dS_minus + self.dS_plus)/2

    @property
    def Fx(self):
        return self.k*(self.tau_plus_x - self.tau_minus_x)/self.dS

    @property
    def Fy(self):
        return self.k*(self.tau_plus_y - self.tau_minus_y)/self.dS
