import matplotlib.pyplot as plt
import numpy as np

from numpy import pi

if __name__ == "__main__":
    L = 5
    H = 5
    n = 35
    X = np.linspace(0, L, n)
    Y = np.linspace(0, H, n)
    xv, yv = np.meshgrid(X, Y)
    x_coef = 4.73
    y_coef = 5
    u = L * np.sin(x_coef * pi * xv / L) * np.cos(y_coef * pi * yv / H)
    v = -H * np.cos(x_coef * pi * xv / L) * np.sin(y_coef * pi * yv / H)
    plt.contourf(xv, yv, u)
    plt.show()
    plt.contourf(xv, yv, v)
    plt.show()
    plt.quiver(xv, yv, u, v)
    plt.show()
