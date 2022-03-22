from itertools import product
from math import floor

import numpy as np


def delta_spread(r, h):
    return (1 / h) * (1 + np.cos(np.pi * r / (2 * h))) / 4


def interp(fluid_x, fluid_y, membrane_x, membrane_y, f):
    dx = 1 / fluid_x[0].size
    dy = 1 / fluid_y[0].size
    F = np.zeros(membrane_x.shape)
    zipped = zip(membrane_x, membrane_y)
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
            Xk = fluid_x[j, i]
            Yk = fluid_y[j, i]
            F[membrane_idx] += (
                fn(Xk, Yk)
                * delta_spread(Xk - xk, dx)
                * delta_spread(Yk - yk, dy)
                * dx
                * dy
            )
    return F


if __name__ == "__main__":
    theta = np.linspace(0, 2 * np.pi, 2000)
    theta = theta[0:-1]
    X = np.cos(theta) / 3 + 1 / 2
    Y = np.sin(theta) / 3 + 1 / 2

    # Define a nonsense force function
    force = lambda x, y: np.sign(x) + np.sign(y)
    f = force(X, Y)

    # Define grid
    L = 1
    H = 1
    Nx = 1000
    Ny = 1000
    dx = L / Nx
    dy = H / Ny
    Xp = np.linspace(dx / 2, L - dx / 2, Nx)
    Yp = np.linspace(dy / 2, H - dy / 2, Ny)

    xv, yv = np.meshgrid(Xp, Yp)

    F = interp(xv, yv, X, Y, f)
