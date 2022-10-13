from dataclasses import dataclass
from math import floor
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm
from solver.stokes import StokesSolver
from solver.ib_solver import Membrane, Fluid, Simulation


if __name__ == '__main__':
    # Define membrane components
    theta = np.linspace(0, 2 * np.pi, 560)
    theta = theta[0:-1]
    X = np.cos(theta) / 3 + 1 / 2
    Y = np.sin(theta) / 3 + 1 / 2

    k = 0.01

    # Define fluid grid
    L = 1
    H = 1
    Nx = 32
    Ny = 32
    dx = L / Nx
    dy = H / Ny
    Xp = np.linspace(dx / 2, L - dx / 2, Nx)
    Yp = np.linspace(dy / 2, H - dy / 2, Ny)
    xv, yv = np.meshgrid(Xp, Yp)

    membrane = Membrane(X, Y, k)
    fluid = Fluid(xv, yv)
    fluid.register(membrane)

    # Circular membrane test case
    X = 0.5 + np.cos(theta)
    Y = 0.5 + np.sin(theta)

    membrane = Membrane(X, Y, k)
    fluid = Fluid(xv, yv)
    fluid.register(membrane)

    dt = 0.01
    simulation = Simulation(fluid, membrane, dt)

    t = 0.0
    import time
    s0 = time.time()
    while t < 10*dt:
        print(f"{t=}")
        step = simulation.step()
        step.plot_pressure()
        t += dt
    s1 = time.time()
    print(f"{s1 - s0:,}s")

