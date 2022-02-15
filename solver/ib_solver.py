import random

from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

class Membrane:

    def __init__(self, x, y):
        self.x_raw = x
        self.y_raw = y

    @property
    def zipped(self):
        return np.array(list(zip(self.x_raw, self.y_raw)), dtype=[('x', float), ('y', float)])


    @property
    def x(self):
        return self.zipped['x']

    @property
    def y(self):
        return self.zipped['y']


    def update_positions(self, new_x, new_y):
        # This is a little awkward -- I think in practice we might use actual getter
        # and setter methods here, which would enable us to cache the properties properly
        # rather than dynamically calculating them each time by zipping x and y
        self.x_raw = new_x
        self.y_raw = new_y


class IBSolver:

    def __init__(self, membrane: Membrane):
        self.membrane = membrane
        self.t = 0


    def step_position(self, x, y):
        return x + np.random.uniform(0, 0.5, x.shape)*(-1)**self.t, y + np.random.uniform(0, 0.5, y.shape)*(-1)**self.t


    def update_membrane_position(self):
        # This might be more appropriate as a method belonging to the Membrane class
        # The Membrane constructor might instead take a position_fn parameter which
        # is called on each time step rather than being called by the IBSolver
        new_x, new_y = self.step_position(
            self.membrane.x,
            self.membrane.y
        )
        self.membrane.update_positions(new_x, new_y)
        self.t += 1


    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.membrane.x, self.membrane.y, '--o')
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.grid()
        ax.set_title(f"t={self.t}")
        fig.savefig(f"Plots/{self.t:03}.png")


if __name__ == '__main__':
    m = Membrane(
        [1, 1.5, 2, 2.5, 3, 3.5, 4],
        [1, 1.5, 1.5, 2.5, 2.5, 3, 5.5]
    )
    s = IBSolver(m)

    for i in range(200):
        s.update_membrane_position()
        s.plot()
