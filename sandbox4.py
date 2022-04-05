from dataclasses import dataclass
from math import floor
from itertools import product

import numpy as np

from scipy.linalg import norm

from solver.stokes import StokesSolver


def delta_spread(r, h):
    return (1 / h) * (1 + np.cos(np.pi * r / (2 * h))) / 4


class Simulation:

    def __init__(self, fluid, membrane, dt, mu=1):
        self.fluid = fluid
        self.membrane = membrane
        self.dt = dt
        self.mu = mu

    def calculate_forces(self):
        return self.membrane.Fx, self.membrane.Fy

    def spread_forces(self, Fx, Fy):
        return self.fluid.spread(Fx), self.fluid.spread(Fy)

    def calculate_velocities(self, fx, fy):
        u, v = self.fluid.stokes_solve(fx, fy, self.mu)
        return self.membrane.interp(u), self.membrane.interp(v)

    def update_membrane_positions(self, U, V):
        self.membrane.X += self.dt*U
        self.membrane.Y += self.dt*V

    def step(self):
        Fx, Fy = self.calculate_forces()
        fx, fy = self.spread_forces(Fx, Fy)
        U, V = self.calculate_velocities(fx, fy)
        self.update_membrane_positions(U, V)


class Fluid:

    def __init__(self, xv, yv, membrane=None):
        self.xv = xv
        self.yv = yv
        self.membrane = membrane

    def register(self, membrane):
        self.membrane = membrane
        self.membrane.fluid = self

    def stokes_solve(self, fx, fy, mu=1):
        solver = StokesSolver(self.xv, self.yv, fx, fy, mu=mu)
        return solver.u, solver.v

    def spread(self, F):
        dx = 1 / self.xv[0].size
        dy = 1 / len(self.yv)
        f = np.zeros(self.xv.shape)
        for xk, yk, ds, Fk in zip(self.membrane.X, self.membrane.Y, self.membrane.dS, F):
            xk_dx = (xk - dx / 2) / dx
            yk_dy = (yk - dy / 2) / dy
            floored_x = floor(xk_dx)
            floored_y = floor(yk_dy)
            # If floored_x == xk_dx, then floored_x - 1 will only
            # capture 1 gridpoint to the left of xk rather than 2
            # Likewise for floored_y and yk_dy
            correction_term_x = -1 if floored_x == xk_dx else 0
            correction_term_y = -1 if floored_y == yk_dy else 0
            x_range = range(
                floored_x - 1 + correction_term_x, floored_x + 3 + correction_term_x
            )
            y_range = range(
                floored_y - 1 + correction_term_y, floored_y + 3 + correction_term_y
            )
            i = floored_x - 1 + correction_term_x
            for i, j in product(x_range, y_range):
                Xk = self.xv[j, i]
                Yk = self.yv[j, i]
                f[j, i] += Fk * delta_spread(Xk - xk, dx) * delta_spread(Yk - yk, dy) * ds
        return f



class Membrane:

    def __init__(self, X, Y, k, fluid=None, p=2):
        self.X = X
        self.Y = Y
        self.k = k
        self.p = p
        self.fluid = fluid


    def interp(self, f):
        if self.fluid is None:
            helpful_message = """Membrane has not been registered to any fluid; cannot interp
            Create a Fluid object and call Fluid.register(membrane) before interpolating forces
            """
            raise Exception(helpful_message)
        dx = 1 / self.fluid.xv[0].size
        dy = 1 / len(self.fluid.yv)
        F = np.zeros(self.X.shape)
        zipped = zip(self.X, self.Y)
        for membrane_idx, xk_yk in enumerate(zipped):
            xk, yk = xk_yk
            xk_dx = (xk - dx / 2) / dx
            yk_dy = (yk - dy / 2) / dy
            floored_x = floor(xk_dx)
            floored_y = floor(yk_dy)
            # If floored_x == xk_dx, then floored_x - 1 will only
            # capture 1 gridpoint to the left of xk rather than 2
            # Likewise for floored_y and yk_dy
            correction_term_x = -1 if floored_x == xk_dx else 0
            correction_term_y = -1 if floored_y == yk_dy else 0
            x_range = range(
                floored_x - 1 + correction_term_x, floored_x + 3 + correction_term_x
            )
            y_range = range(
                floored_y - 1 + correction_term_y, floored_y + 3 + correction_term_y
            )
            i = floored_x - 1 + correction_term_x
            for i, j in product(x_range, y_range):
                Xk = self.fluid.xv[j, i]
                Yk = self.fluid.yv[j, i]
                F[membrane_idx] += (
                    f[j, i]
                    * delta_spread(Xk - xk, dx)
                    * delta_spread(Yk - yk, dy)
                    * dx
                    * dy
                )
        return F



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


if __name__ == '__main__':
    # Define membrane components
    theta = np.linspace(0, 2 * np.pi, 2000)
    theta = theta[0:-1]
    X = np.cos(theta) / 3 + 1 / 2
    Y = np.sin(theta) / 3 + 1 / 2

    # Define a nonsense force function
    force = lambda x, y: np.sign(x) + np.sign(y)
    k = 0.01

    # Define fluid grid
    L = 1
    H = 1
    Nx = 1000
    Ny = 1000
    dx = L / Nx
    dy = H / Ny
    Xp = np.linspace(dx / 2, L - dx / 2, Nx)
    Yp = np.linspace(dy / 2, H - dy / 2, Ny)
    xv, yv = np.meshgrid(Xp, Yp)

    # Instantiate membrane and fluid
    membrane = Membrane(X, Y, k)
    fluid = Fluid(xv, yv)
    fluid.register(membrane)

    dt = 0.01
    simulation = Simulation(fluid, membrane, dt)

    t_end = 0.05
    t = 0.0
    while t < t_end:
        print(f"{t=}")
        simulation.step()
        t += dt
