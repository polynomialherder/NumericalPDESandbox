import distmesh as dm
import matplotlib.pyplot as plt
import numpy as np


def meshplot(p, t, write_location=None):
    fig, ax = plt.subplots()
    # PyDistMesh appears to access this deprecated property
    # from matplotlib.Axes, so we initialize it to keep it from
    # throwing an exception
    ax._hold = True
    dm.plotting.axes_simpplot2d(ax, p, t)
    if not write_location:
        fig.show()
    else:
        fig.savefig(write_location)


def circle_mesh(r: float, q: np.ndarray, h=0.2):
    def distance(x):
        return np.sqrt(((x - q)**2).sum(1)) - r

    bbox = (q[0] - r, q[1] - r, q[0] + r, q[1] + r)
    return dm.distmesh2d(distance, dm.huniform, h, bbox)


if __name__ == '__main__':
    points, connections = circle_mesh(2.5, np.array([3, 4]))
    meshplot(points, connections)

    # Poor man's meshplot
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], "o")
    ax.grid()
    fig.show()
