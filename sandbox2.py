import timeit

from itertools import product
from math import floor

import numpy as np
import matplotlib.pyplot as plt


def arc_length_left(x, y):

    xkm1 = np.roll(x, -1)
    ykm1 = np.roll(y, -1)

    return np.sqrt((xkm1 - x) ** 2 + (ykm1 - y) ** 2)


def arc_length_right(x, y):
    # Shift elements in both arrays to the left by 1 unit
    # for differencing
    xkp1 = np.roll(x, 1)
    ykp1 = np.roll(y, 1)

    return np.sqrt((xkp1 - x) ** 2 + (ykp1 - y) ** 2)


def arc_length(x, y):
    return (arc_length_left(x, y) + arc_length_right(x, y)) / 2


def delta_spread(r, h):
    return (1 / h) * (1 + np.cos(np.pi * r / (2 * h))) / 4


def spread_mask(fluid_x, fluid_y, membrane_x, membrane_y, fn):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1 / fluid_x[0].size
    dy = 1 / fluid_y[0].size
    F = fn(membrane_x, membrane_y)
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
        diff_x_all = fluid_x - xk
        diff_y_all = fluid_y - yk
        abs_diff_x = np.abs(diff_x_all)
        abs_diff_y = np.abs(diff_y_all)
        abs_diff_x_lt_2dx = abs_diff_x < 2 * dx
        abs_diff_y_lt_2dy = abs_diff_y < 2 * dy
        mask = np.logical_and(abs_diff_x_lt_2dx, abs_diff_y_lt_2dy)
        diff_x = diff_x_all[mask]
        diff_y = diff_y_all[mask]
        f[mask] += Fk * delta_spread(diff_x, dx) * delta_spread(diff_y, dy) * ds
    return f


def spread_while(fluid_x, fluid_y, membrane_x, membrane_y, F):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1 / fluid_x[0].size
    dy = 1 / fluid_y[0].size
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
        xk_dx = (xk - dx / 2) / dx
        yk_dy = (yk - dy / 2) / dy
        floored_x = floor(xk_dx)
        floored_y = floor(yk_dy)
        # If floored_x == xk_dx, then floored_x - 1 will only
        # capture 1 gridpoint to the left of xk rather than 2
        # Likewise for floored_y and yk_dy
        correction_term_x = -1 if floored_x == xk_dx else 0
        correction_term_y = -1 if floored_y == yk_dy else 0
        end_x = floored_x + 3 + correction_term_x
        start_y = floored_y - 1 + correction_term_y
        end_y = floored_y + 3 + correction_term_y
        i = floored_x - 1 + correction_term_x
        while i < end_x:
            j = start_y
            while j < end_y:
                Xk = fluid_x[j, i]
                Yk = fluid_y[j, i]
                f[j, i] += (
                    Fk * delta_spread(Xk - xk, dx) * delta_spread(Yk - yk, dy) * ds
                )
                j += 1
            i += 1
    return f


def spread_unrolled(fluid_x, fluid_y, membrane_x, membrane_y, fn):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1 / fluid_x[0].size
    dy = 1 / fluid_y[0].size
    F = fn(membrane_x, membrane_y)
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
        xk_dx = (xk - dx / 2) / dx
        yk_dy = (yk - dy / 2) / dy
        floored_x = floor(xk_dx)
        floored_y = floor(yk_dy)
        # If floored_x == xk_dx, then floored_x - 1 will only
        # capture 1 gridpoint to the left of xk rather than 2
        # Likewise for floored_y and yk_dy
        correction_term_x = -1 if floored_x == xk_dx else 0
        correction_term_y = -1 if floored_y == yk_dy else 0
        j = floored_y - 1 + correction_term_y
        i = floored_x - 1 + correction_term_x
        ip1 = i + 1
        ip2 = i + 2
        ip3 = i + 3
        jp1 = j + 1
        jp2 = j + 2
        jp3 = j + 3
        f[j, i] += (
            Fk
            * delta_spread(fluid_x[j, i] - xk, dx)
            * delta_spread(fluid_y[j, i] - yk, dy)
            * ds
        )
        f[jp1, i] += (
            Fk
            * delta_spread(fluid_x[jp1, i] - xk, dx)
            * delta_spread(fluid_y[jp1, i] - yk, dy)
            * ds
        )
        f[jp2, i] += (
            Fk
            * delta_spread(fluid_x[jp2, i] - xk, dx)
            * delta_spread(fluid_y[jp2, i] - yk, dy)
            * ds
        )
        f[jp3, i] += (
            Fk
            * delta_spread(fluid_x[jp3, i] - xk, dx)
            * delta_spread(fluid_y[jp3, i] - yk, dy)
            * ds
        )
        f[j, ip1] += (
            Fk
            * delta_spread(fluid_x[j, ip1] - xk, dx)
            * delta_spread(fluid_y[j, ip1] - yk, dy)
            * ds
        )
        f[jp1, ip1] += (
            Fk
            * delta_spread(fluid_x[jp1, ip1] - xk, dx)
            * delta_spread(fluid_y[jp1, ip1] - yk, dy)
            * ds
        )
        f[jp2, ip1] += (
            Fk
            * delta_spread(fluid_x[jp2, ip1] - xk, dx)
            * delta_spread(fluid_y[jp2, ip1] - yk, dy)
            * ds
        )
        f[jp3, ip1] += (
            Fk
            * delta_spread(fluid_x[jp3, ip1] - xk, dx)
            * delta_spread(fluid_y[jp3, ip1] - yk, dy)
            * ds
        )
        f[j, ip2] += (
            Fk
            * delta_spread(fluid_x[j, ip2] - xk, dx)
            * delta_spread(fluid_y[j, ip2] - yk, dy)
            * ds
        )
        f[jp1, ip2] += (
            Fk
            * delta_spread(fluid_x[jp1, ip2] - xk, dx)
            * delta_spread(fluid_y[jp1, ip2] - yk, dy)
            * ds
        )
        f[jp2, ip2] += (
            Fk
            * delta_spread(fluid_x[jp2, ip2] - xk, dx)
            * delta_spread(fluid_y[jp2, ip2] - yk, dy)
            * ds
        )
        f[jp3, ip2] += (
            Fk
            * delta_spread(fluid_x[jp3, ip2] - xk, dx)
            * delta_spread(fluid_y[jp3, ip2] - yk, dy)
            * ds
        )
        f[j, ip3] += (
            Fk
            * delta_spread(fluid_x[j, ip3] - xk, dx)
            * delta_spread(fluid_y[j, ip3] - yk, dy)
            * ds
        )
        f[jp1, ip3] += (
            Fk
            * delta_spread(fluid_x[jp1, ip3] - xk, dx)
            * delta_spread(fluid_y[jp1, ip3] - yk, dy)
            * ds
        )
        f[jp2, ip3] += (
            Fk
            * delta_spread(fluid_x[jp2, ip3] - xk, dx)
            * delta_spread(fluid_y[jp2, ip3] - yk, dy)
            * ds
        )
        f[jp3, ip3] += (
            Fk
            * delta_spread(fluid_x[jp3, ip3] - xk, dx)
            * delta_spread(fluid_y[jp3, ip3] - yk, dy)
            * ds
        )
    return f


def spread(fluid_x, fluid_y, membrane_x, membrane_y, F):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1 / fluid_x[0].size
    dy = 1 / fluid_y[0].size
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
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
            f[j, i] += Fk * delta_spread(Xk - xk, dx) * delta_spread(Yk - yk, dy) * ds
    return f


if __name__ == "__main__":

    theta = np.linspace(0, 2 * np.pi, 2000)
    theta = theta[0:-1]
    X = np.cos(theta) / 3 + 1 / 2
    Y = np.sin(theta) / 3 + 1 / 2

    # Define a nonsense force function
    force = lambda x, y: np.sign(x) + np.sign(y)
    F = force(X, Y)

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

    runs = 100
    f = spread(xv, yv, X, Y, force)
