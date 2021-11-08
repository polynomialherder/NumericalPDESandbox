import math

import numpy as np

from solver.stokes import StokesSolver


if __name__ == '__main__':
    s = StokesSolver(
        f = lambda x: x,
        g = lambda y: y,
        f_actual = lambda x: x,
        g_actual = lambda y: y,
        rows_x = 10,
        rows_y = 10,
        x_lower = 0,
        x_upper = 1,
        y_lower = 0,
        y_upper = 1
    )
