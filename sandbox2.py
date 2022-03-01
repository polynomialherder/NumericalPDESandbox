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
    return (1/h)*(1 + np.cos(np.pi*r/(2*h)))


def spread(fluid_x, fluid_y, membrane_x, membrane_y, fn):
    dS = arc_length(membrane_x, membrane_y)
    dx = 1/fluid_x[0].size
    dy = 1/fluid_y[0].size
    F = fn(fluid_x, fluid_y)
    f = np.zeros(fluid_x.shape)
    for xk, yk, ds in zip(X, Y, dS):
        mask = np.logical_and(abs(fluid_x - xk) < 2*dx, abs(fluid_y - yk) < 2*dy)
        diff_x = (fluid_x - xk)[mask]
        diff_y = (fluid_y - yk)[mask]
        f[mask] += delta_spread(diff_x, dx) * delta_spread(diff_y, dy) * ds
    return F*f



if __name__ == '__main__':
    # Define a rectangular membrane
    X = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
    )
    Y = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.3]
    )

    # Define a nonsense force function
    force = lambda x, y: np.sign(x) + np.sign(y)

    # Define grid
    Xp = np.linspace(0, 1, 20)
    Yp = np.linspace(0, 1, 20)
    xv, yv = np.meshgrid(Xp, Yp)

    f = spread(xv, yv, X, Y, force)

    plt.plot(X, Y, "ro")
    plt.pcolor(xv, yv, f)
    plt.colorbar()

    plt.show()
