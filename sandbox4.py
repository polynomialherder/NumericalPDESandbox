from dataclasses import dataclass

import numpy as np


class Membrane:

    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k

    def left_differences(self, vec):
        shifted = np.roll(vec, -1)
        return shifted - vec

    def right_differences(self, vec):
        shifted = np.roll(vec, 1)
        return shifted - vec

    @property
    def tau_left_x(self):
        return self.left_differences(self.X)

    @property
    def tau_right_x(self):
        return self.right_differences(self.X)

    @property
    def tau_left_y(self):
        return self.left_differences(self.Y)

    @property
    def tau_right_y(self):
        return self.right_differences(self.Y)

    @property
    def tau_x(self):
        return self.tau_left_x + self.tau_right_x

    @property
    def tau_y(self):
        return self.tau_left_y + self.tau_right_y

    @property
    def arc_length_left(self):
        return np.sqrt(self.left_differences(self.X)**2 + self.left_differences(self.Y)**2)

    @property
    def arc_length_right(self):
        return np.sqrt(self.right_differences(self.X)**2 + self.right_differences(self.Y)**2)

    @property
    def dS(self):
        return (self.arc_length_left + self.arc_length_right)/2

    @property
    def Fx(self):
        return self.k*(self.tau_left_x - self.tau_right_x)/self.dS

    @property
    def Fy(self):
        return self.k*(self.tau_left_y - self.tau_right_y)/self.dS


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
