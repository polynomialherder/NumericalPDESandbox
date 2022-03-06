from itertools import product
from math import floor

import numpy as np
import matplotlib.pyplot as plt

def arc_length_left(x, y):

    xkm1 = np.roll(x, -1)
    ykm1 = np.roll(y, -1)

    return np.sqrt((xkm1 - x)**2 + (ykm1 - y)**2)

def arc_length_right(x, y):
    # Shift elements in both arrays to the left by 1 unit
    # for differencing
    xkp1 = np.roll(x, 1)
    ykp1 = np.roll(y, 1)

    return np.sqrt((xkp1 - x)**2 + (ykp1 - y)**2)

def arc_length(x, y):
    return (arc_length_left(x, y) + arc_length_right(x, y))/2

def delta_spread(r, h):
    return (1/h)*(1 + np.cos(np.pi*r/(2*h)))/4


#@profile
def spread(fluid_x, fluid_y, membrane_x, membrane_y, fn):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1/fluid_x[0].size
    dy = 1/fluid_y[0].size
    F = fn(membrane_x, membrane_y)
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
        diff_x_all = fluid_x - xk
        diff_y_all = fluid_y - yk
        abs_diff_x = np.abs(diff_x_all)
        abs_diff_y = np.abs(diff_y_all)
        abs_diff_x_lt_2dx = abs_diff_x < 2*dx
        abs_diff_y_lt_2dy = abs_diff_y < 2*dy
        mask = np.logical_and(abs_diff_x_lt_2dx, abs_diff_y_lt_2dy)
        diff_x = diff_x_all[mask]
        diff_y = diff_y_all[mask]
        f[mask] += Fk*delta_spread(diff_x, dx) * delta_spread(diff_y, dy) * ds
    return f

#@profile
def spread_efficient(fluid_x, fluid_y, membrane_x, membrane_y, fn):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1/fluid_x[0].size
    dy = 1/fluid_y[0].size
    F = fn(membrane_x, membrane_y)
    f = np.zeros(fluid_x.shape)
    check = np.zeros(fluid_x.shape)
    for xk, yk, ds, Fk in zip(membrane_x, membrane_y, dS, F):
        xk_dx = (xk-dx/2)/dx
        yk_dy = (yk-dy/2)/dy
        floored_x = floor(xk_dx)
        floored_y = floor(yk_dy)
        # If floored_x == xk_dx, then floored_x - 1 will only
        # capture 1 gridpoint to the left of xk rather than 2
        # Likewise for floored_y and yk_dy
        correction_term_x = -1 if floored_x == xk_dx else 0
        correction_term_y = -1 if floored_y == yk_dy else 0
        x_range = range(
            floored_x - 1 + correction_term_x,
            floored_x + 3 + correction_term_x
        )
        y_range = range(
            floored_y - 1 + correction_term_y,
            floored_y + 3 + correction_term_y
        )
        mask = np.logical_and(
            abs(fluid_x - xk) < 2*dx,
            abs(fluid_y - yk) < 2*dy
        )
        mask_indices = set(zip(*np.where(mask)))
        test = set(zip(fluid_x[mask], fluid_y[mask]))
        test2_indices = list(product(x_range, y_range))
        test2 = set([(fluid_x[j, i], fluid_y[j, i]) for i, j in test2_indices])
        check_range_y = Yp[floored_y - 1 + correction_term_y:floored_y + 3 + correction_term_y]
        check_range_x = Xp[floored_x - 1 + correction_term_x:floored_x + 3 + correction_term_x]
        check_range_y2 = set(fluid_y[mask])
        check_range_x2 = set(fluid_x[mask])
        min_x2 = min(x[0] for x in test2)
        min_y2 = min(y[1] for y in test2)
        diff = test - test2
        diff2 = test2 - test
        testl = list(test)
        test2l = list(test2)
        common = set.intersection(test, test2)
        diff_x = fluid_x[mask] - xk
        diff_y = fluid_y[mask] - yk
        check[mask] += Fk*delta_spread(diff_x, dx)*delta_spread(diff_y, dy)*ds
        if len(list(product(x_range, y_range))) != 16:
            print(f"{x_range} x {y_range} does not contain 16 grid points")
            print(f"{xk=}, {dx=}, {yk=}, {dy=}")
        for i, j in product(x_range, y_range):
            Xk = fluid_x[j, i]
            Yk = fluid_y[j, i]
            f[j, i] += Fk*delta_spread(Xk - xk, dx)*delta_spread(Yk - yk, dy)*ds
            if not any(np.isclose(f[j, i], check[mask])):
                breakpoint()
    return f



if __name__ == '__main__':

    theta = np.linspace(0, 2*np.pi, 10)
    theta = theta[0:-1]
    X = np.cos(theta)/3 + 1/2
    Y = np.sin(theta)/3 + 1/2

    # Define a nonsense force function
    force = lambda x, y: np.sign(x) + np.sign(y)

    # Define grid
    L = 1
    H = 1
    Nx = 20
    Ny = 20
    dx = L/Nx
    dy = H/Ny
    Xp = np.linspace(dx/2, L-dx/2, Nx)
    Yp = np.linspace(dy/2, H-dy/2, Ny)

    xv, yv = np.meshgrid(Xp, Yp)

    import time
    start_time = time.time()
    print(f"Calculating f using masks")
    f = spread(xv, yv, X, Y, force)
    end_time = time.time()
    print(f"Finished calculating f using masks, took {end_time - start_time}s")

    # plt.plot(X, Y, "ro")
    # plt.pcolor(xv, yv, f)
    # plt.colorbar()

    # plt.show()

    ds = arc_length(X, Y)
    fmemb = force(X, Y)
    membint = sum(fmemb*ds)
    gridint = sum(sum(f))*dx*dy


    print(f"Calculating f using index arithmetic")
    start_time = time.time()
    f2 = spread_efficient(xv, yv, X, Y, force)
    end_time = time.time()
    print(f"Finished calculating f using index arithmetic, took {end_time - start_time}s")
