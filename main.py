from dataclasses import dataclass
from itertools import product
import logging
from math import floor

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm
from solver.stokes import StokesSolver
from solver.ib_solver import Membrane, Fluid, Simulation

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True
logging.getLogger('asyncio').disabled = True



if __name__ == '__main__':
    # Define membrane components
    theta = np.linspace(0, 2*np.pi, 560)
    theta = theta[0:-1]

    k = 0.01

    # Define fluid grid
    L = 1
    H = 1
    Nx = 128
    Ny = 128
    dx = L / Nx
    dy = H / Ny
    Xp = np.linspace(dx / 2, L - dx / 2, Nx)
    Yp = np.linspace(dy / 2, H - dy / 2, Ny)
    xv, yv = np.meshgrid(Xp, Yp)

    # Circular membrane test case
    X = 0.5 + (1/3)*np.cos(theta)
    Y = 0.5 + (1/3)*np.sin(theta)

    membrane = Membrane(X, Y, k)
    fluid = Fluid(xv, yv)
    fluid.register(membrane)

    dt = 0.01
    simulation = Simulation(fluid, membrane, dt)
    simulation.perform_simulation(write_format="csv")
