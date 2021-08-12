import math

import numpy as np

from numpy.fft import fft2, fftshift, ifft2

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver
from solver.second_order import _2norm

class PoissonSolver2D:

    def __init__(self, f, rows, x_lower, x_upper, y_lower, y_upper):
        self.f = f
        self.rows = rows
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper
        # Note that x and y are initialized and refreshed when self.F is
        # invoked
        self.x = None
        self.y = None

    @property
    def length_x(self):
        return self.x_upper - self.x_lower

    @property
    def h(self):
        return self.length_x/self.rows

    @property
    def length_y(self):
        return self.y_upper - self.y_lower

    @property
    def l(self):
        return self.length_y/self.rows

    @property
    def _x(self):
        return np.linspace(self.x_lower + self.h/2, self.x_upper - self.h/2, self.rows, endpoint=True)

    @property
    def _y(self):
        return np.linspace(self.y_lower + self.l/2, self.y_upper - self.l/2, self.rows, endpoint=True)

    @property
    def meshgrid(self):
        return np.meshgrid(self._x, self._y)

    @property
    def F(self):
        self.x, self.y = self.meshgrid
        return self.f(self.x, self.y)


    @property
    def coefficients(self):
        midpoint = self.rows // 2
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        row_indices = (i if i <= midpoint else (self.rows-i) for i in range(self.rows))
        denominator_term_x = lambda n: (2*math.pi*n/self.length_x)**2
        denominator_term_y = lambda n: (2*math.pi*n/self.length_y)**2
        coefficients = []
        # TODO: We don't actually need to calculate j for each iteration here, since the value for j
        #       will be the same for the given column index. We should cache the values for j to avoid
        #       unnecessary recalculation
        for row_index in row_indices:
            row = []
            for column_index in range(self.rows):
                if not row_index and not column_index:
                    row.append(0)
                    continue
                elif column_index <= midpoint:
                    j = column_index
                else:
                    j = self.rows - column_index
                i = row_index
                denominator = denominator_term_y(i) + denominator_term_x(j)
                row.append(-1/denominator)
            coefficients.append(
                np.fromiter(row, dtype=float)
            )
        return np.array(coefficients)

    @property
    def transformed(self):
        return fft2(self.F)

    @property
    def ifft(self):
        return ifft2(self.transformed*self.coefficients)

    @property
    def U(self):
        return np.real(self.ifft)



if __name__ == '__main__':
    # This lambda is just placeholder value until I come up with a valid test case
    f = lambda x, y: -4*np.sin(2*x*y)

    # Initialize a 2D solver
    p = PoissonSolver2D(f, 10, 0, 1, 0, 25)

    # Inspect the complex solution (inverse fft2 of fft2(F)*fourier_coefficients prior to coercion to a real-valued array)
    complex_solution = p.ifft

    # Inspect the real solution
    real_solution = p.U
