import numpy as np

from itertools import product
from math import floor

from numba import njit

def delta_spread(r, h):
    return (1 / h) * (1 + np.cos(np.pi * r / (2 * h))) / 4


def spread_to_fluid(F, fluid, membrane):
    xv = fluid.xv
    yv = fluid.yv
    X = membrane.X
    Y = membrane.Y
    dS = membrane.delta_theta
    return spread(F, xv, yv, X, Y, dS)


def spread(F, xv, yv, X, Y, dS):
    dx = 1 / xv[0].size
    dy = 1 / len(yv)
    xlen, ylen = xv.shape
    f = np.zeros((xlen, ylen), dtype=np.float64)
    for xk, yk, ds, Fk in zip(X, Y, dS, F):
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
        for i in x_range:
            im = i % xlen
            for j in y_range:
                jm = j % ylen
                Xk = xv[jm, im]
                Yk = yv[jm, im]
                rx = Xk - xk
                ry = Yk - yk
                delta_spread_x = (1 / dx) * (1 + np.cos(np.pi * rx / (2 * dx))) / 4
                delta_spread_y = (1 / dy) * (1 + np.cos(np.pi * ry / (2 * dy))) / 4
                f[jm, im] += Fk * delta_spread_x * delta_spread_y * ds
    return f


def interp_to_membrane(f, fluid, membrane):
    return interp(f, fluid.xv, fluid.yv, membrane.X, membrane.Y, membrane.delta_theta)


def interp(f, xv, yv, X, Y, dS):
    dx = 1 / xv[0].size
    dy = 1 / len(yv)
    F = np.zeros(X.shape)
    zipped = zip(X, Y)
    xlen, ylen = xv.shape
    for idx, xk_yk in enumerate(zipped):
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
        for i in x_range:
            im = i % xlen
            for j in y_range:
                jm = j % ylen
                Xk = xv[jm, im]
                Yk = yv[jm, im]
                rx = Xk - xk
                ry = Yk - yk
                delta_spread_x = (1 / dx) * (1 + np.cos(np.pi * rx / (2 * dx))) / 4
                delta_spread_y = (1 / dy) * (1 + np.cos(np.pi * ry / (2 * dy))) / 4
                F[idx] += f[jm, im] * delta_spread_x * delta_spread_y * dx * dy
    return F
