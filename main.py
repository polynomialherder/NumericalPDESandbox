import math

import numpy as np

from solver.stokes import StokesSolver

if __name__ == '__main__':
    s = StokesSolver(
        f = lambda x, y: x+y,
        g = lambda x, y: 2*(y+x**2),
        f_actual = lambda x, y: x+y,
        g_actual = lambda x, y: 2*(y+x**2),
        rows_x = 10,
        rows_y = 10,
        x_lower = 0,
        x_upper = 1,
        y_lower = 0,
        y_upper = 1
    )
