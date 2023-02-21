from dataclasses import dataclass
from itertools import product
import logging
import warnings

from math import floor

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm
from solver.stokes import StokesSolver
from solver.ib_solver import Membrane, Fluid, Simulation

warnings.filterwarnings("ignore", module="matplotlib")

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True
logging.getLogger('asyncio').disabled = True


if __name__ == '__main__':
    # Define membrane components
    theta = np.linspace(0, 2*np.pi, 560)
    theta = theta[0:-1]

    # Define the Hooke's constant
    k = 0.5

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
    X = 0.5 + (1/8)*np.cos(theta)
    Y = 0.5 + (1/4)*np.sin(theta)

    X_ref = 0.5 + (1/32)*np.cos(theta)
    Y_ref = 0.5 + (1/32)*np.sin(theta)

    membrane = Membrane(X, Y, X_ref=X_ref, Y_ref=Y_ref, k=k)
    fluid = Fluid(xv, yv)
    fluid.register(membrane)

    # dt=0.18, mu=0.3, k=1.5
    dt = 0.00001
    simulation = Simulation(fluid, membrane, dt, mu=0.3, save_history=True)
    simulation.perform_simulation(iterations=5500, data_format="csv", image_format="png", write_frequency=5, plot_frequency=5)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(simulation.membrane.X, simulation.history[-1].Fx)
    ax[1].plot(simulation.membrane.Y, simulation.history[-1].Fy)
    fig.show()
