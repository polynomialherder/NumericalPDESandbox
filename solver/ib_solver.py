import random

from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from solver.stokes import StokesSolver

class ImmersedBoundary:

    def __init__(self, x, y):
        self.x_raw = x
        self.y_raw = y
        self.fluid = None

    @property
    def zipped(self):
        return np.array(list(zip(self.x_raw, self.y_raw)), dtype=[('x', float), ('y', float)])


    @property
    def x(self):
        return self.zipped['x']

    @property
    def y(self):
        return self.zipped['y']

    @property
    def ds_left(self):
        xkm1 = np.roll(self.x, -1)
        ykm1 = np.roll(self.y, -1)
        return np.sqrt((xkm1 - self.x)**2 + (ykm1 - self.y)**2)

    @property
    def ds_right(self):
        xkp1 = np.roll(self.x, -1)
        ykp1 = np.roll(self.y, -1)
        return np.sqrt((xkp1 - self.x)**2 + (ykp1 - self.y)**2)

    @property
    def dS(self):
        return (self.ds_left + self.ds_right)/2

    @staticmethod
    def delta_spread(r, h):
        if abs(r) < 2:
            return 0
        return (1/h)*(1 + np.cos(np.pi*r/(2*h)))


    def spread(self, fn):
        dx = self.fluid.dx
        dy = self.fluid.dy
        F = fn(self.fluid.xv, self.fluid.yv)
        f = np.zeros(self.fluid.xv.shape)
        for xk, yk, ds in zip(self.x, self.y, self.dS):
            mask = np.logical_and(
                abs(self.fluid.xv - xk) < 2*self.fluid.dx,
                abs(self.fluid.yv - yk) < 2*self.fluid.dy
            )
            diff_x = (self.fluid.xv - xk)[mask]
            diff_y = (self.fluid.yv - yk)[mask]
            f[mask] += self.delta_spread(diff_x, self.fluid.dx) * self.delta_spread(diff_y, self.fluiddy) * ds
        return F*f



    def calculate_forces(self):
        pass


    def update_positions(self, new_x, new_y):
       # This is a little awkward -- I think in practice we might use actual getter
        # and setter methods here, which would enable us to cache the properties properly
        # rather than dynamically calculating them each time by zipping x and y
        self.x_raw = new_x
        self.y_raw = new_y


class Fluid:

    def __init__(self, X, Y):
        self.t = 0
        self.X = X
        self.Y = Y
        self._xv = None
        self._yv = None
        self.dx = 1/self.X.size
        self.dy = 1/self.Y.size

    def register(self, immersed_boundary: ImmersedBoundary):
        self.immersed_boundary = immersed_boundary
        immersed_boundary.fluid = self

    @property
    def meshgrid(self):
        return np.meshgrid(self.X, self.Y)

    @property
    def xv(self):
        if self._xv is None:
            self._xv, self._yv = self.meshgrid
        return self._xv

    @property
    def yv(self):
        if self._yv is None:
            self._xv, self._yv = self.meshgrid
        return self._yv


    def step_position(self, x, y):
        return x + np.random.uniform(0, 0.5, x.shape)*(-1)**self.t, y + np.random.uniform(0, 0.5, y.shape)*(-1)**self.t


    def update_immersed_boundary_position(self):
        # This might be more appropriate as a method belonging to the Immersed_Boundary class
        # The Immersed_Boundary constructor might instead take a position_fn parameter which
        # is called on each time step rather than being called by the IBSolver
        new_x, new_y = self.step_position(
            self.immersed_boundary.x,
            self.immersed_boundary.y
        )
        self.immersed_boundary.update_positions(new_x, new_y)
        self.t += 1


    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.immersed_boundary.x, self.immersed_boundary.y, '--o')
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.grid()
        ax.set_title(f"t={self.t}")
        fig.savefig(f"Plots/{self.t:03}.png")


if __name__ == '__main__':
    membrane_X = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
    )
    membrane_Y = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.3]
    )
    m = ImmersedBoundary(
        membrane_X, membrane_Y
    )

    X = np.linspace(0, 1, 1000)
    Y = np.linspace(0, 1, 1000)
    fluid = Fluid(X, Y)
    fluid.register(m)

    force = lambda x, y: np.sign(x) + np.sign(y)
    f = m.spread(force)
    plt.plot(membrane_X, membrane_Y, "ro")
    plt.pcolor(fluid.xv, fluid.yv, f)
    plt.colorbar()
    plt.show()
