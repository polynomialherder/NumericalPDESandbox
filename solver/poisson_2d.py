import math

import numpy as np

from numpy.fft import fft2, ifft2


class PoissonSolver2D:

    def __init__(self, f, actual, rows_x, rows_y, x_lower, x_upper, y_lower, y_upper):
        self.f = f
        self.rows_x = rows_x
        self.rows_y = rows_y
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper
        # Note that x and y are initialized and refreshed when self.F is
        # invoked
        self.x = None
        self.y = None
        self.actual = actual

    @property
    def length_x(self):
        return self.x_upper - self.x_lower

    @property
    def h(self):
        return self.length_x/self.rows_x

    @property
    def length_y(self):
        return self.y_upper - self.y_lower

    @property
    def l(self):
        return self.length_y/self.rows_y

    @property
    def domain_area(self):
        return self.length_x * self.length_y

    @property
    def _x(self):
        return np.linspace(self.x_lower + self.h/2, self.x_upper - self.h/2, self.rows_x, endpoint=True)

    @property
    def _y(self):
        return np.linspace(self.y_lower + self.l/2, self.y_upper - self.l/2, self.rows_y, endpoint=True)

    @property
    def meshgrid(self):
        return np.meshgrid(self._x, self._y)

    @property
    def F(self):
        self.x, self.y = self.meshgrid
        return self.f(self.x, self.y)


    def apply_actual(self):
        self.x, self.y = self.meshgrid
        return self.actual(self.x, self.y)


    @property
    def coefficients(self):
        """ Compute the Fourier coefficients based on the scipy fft implementation

        At a high level, this method works in the following way:
        1. The row indices are computed (see the explanation below)
        2. For each row index in the list of row indices, calculate the row of coefficients.
           This entails computing i -- which is always equal to the row index -- and j, which
           is computed in the same manner that the row indices were computed.
        3. The Fourier coefficient for the i,jth term is then calculated. This coefficient is of
           the form -1/(d_i + d_j), where d_i is the denominator term that depends on i, and likewise 
           for d_j.
        4. Transform the row into an array, and append it to an accumulator
        5. Return the accumulator as an array

        Note that this method performs some unnecessary extra computation -- see TODO item below
        """
        midpoint_x = self.rows_x // 2
        midpoint_y = self.rows_y // 2
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        row_indices = (i if i <= midpoint_y else (self.rows_y-i) for i in range(self.rows_y))
        denominator_term_x = lambda n: (2*math.pi*n/self.length_x)**2
        denominator_term_y = lambda n: (2*math.pi*n/self.length_y)**2
        coefficients = []
        # TODO: We don't actually need to calculate j for each iteration here, since the value for j
        #       will be the same for the given column index. We should refactor this so that j is
        #       computed only once.
        for row_index in row_indices:
            row = []
            for column_index in range(self.rows_x):
                if not row_index and not column_index:
                    row.append(0)
                    continue
                elif column_index <= midpoint_x:
                    j = column_index
                else:
                    j = self.rows_x - column_index
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
    def complex_solution(self):
        return ifft2(self.transformed*self.coefficients)

    @property
    def U(self):
        return np.real(self.complex_solution)

    @property
    def solution(self):
        """ Synonymous with PoissonSolver2D.U """
        return self.U

    @property
    def error(self):
        return abs(self.U - self.apply_actual())

    @property
    def twonorm(self):
        return self.pnorm(2)

    def pnorm(self, p):
        return (sum(sum(self.error**2))*self.h*self.l)**1/p
