import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from solver.ib_solver import Simulation
from solver.fluid import Fluid
from solver.membrane import Membrane


if __name__ == '__main__':
    # Define membrane components
    theta = np.linspace(0, 2*np.pi, 560)
    theta = theta[0:-1]

    # Define the Hooke's constant
    k = 0.5
    mu = 0.5

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

    X_ref = 0.5 + (1/np.sqrt(32))*np.cos(theta)
    Y_ref = 0.5 + (1/np.sqrt(32))*np.sin(theta)


    # dt=0.18, mu=0.3, k=1.5
    data = []
    k = 1
    test_dts = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    for mu in test_dts:
        print(f"Locating minimum test dt such that the simulation 'blows up' for {k=}, {mu=}, {test_dts=}")
        membrane = Membrane(X, Y, X_ref=X_ref, Y_ref=Y_ref, k=1)
        fluid = Fluid(xv, yv, mu=mu)
        fluid.register(membrane)
        for dt in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
            print(f"Checking {dt=}")
            with Simulation(fluid, membrane, dt, save_history=True, iterations=5, write=False) as s:
                s.perform_simulation()
                max_membrane = np.max(np.abs(s.membrane.X))
                if max_membrane > 1e1:
                    print(f"Minimum test dt such that the simulation 'blows up' for {k=} and {mu=} is {dt=}")
                    data.append({"dt": dt, "k": k, "mu": mu, "mu/k": mu/k})
                    break
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.plot(df["mu/k"], df.dt, "--")
    ax.set_xlabel("$\\mu/k$")
    ax.set_ylabel("dt")
    fig.suptitle("$\\mu/k$ vs $dt$")
    ax.grid()
    fig.show()
