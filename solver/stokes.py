"""
This module consists of abstractions for solving a simple linear second order differential equation of the form
f(x) = u''(x). The methods implemented are based on the discussion in Finite Difference Methods for Ordinary and Partial
Differential Equations by Randall Leveque, sections 2.1-2.10.
"""
import math

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, lsqr, svds

from solver.boundary import BCType, BoundaryCondition
from solver.second_order_2d import PoissonSolver

MACHINE_EPSILON = 2e-16

class StokesSolver:

    def __init__(self, f, g, f_actual, g_actual, rows_x, rows_y, x_lower, x_upper, y_lower, y_upper, alpha, beta, rows):
        self.f = f
        self.g = g
        self.rows_x = rows_x
        self.rows_y = rows_y
        self.f_actual = f_actual
        self.g_actual = g_actual
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        self.beta = beta
        self.rows = rows


    @property
    def midpoint_x(self):
        return self.rows_x // 2

    @property
    def midpoint_y(self):
        return self.rows_y // 2

    @property
    def indices_x(self):
        return np.array([i if i <= midpoint_x else -(self.rows_x - i) for i in range(self.rows_x)])

    @property
    def indices_y(self):
        return np.array([i if i <= midpoint_y else -(self.rows_y - i) for i in range(self.rows_y)])

    @property
    def length_x(self):
        return self.x_upper - self.x_lower

    @property
    def length_y(self):
        return self.y_upper - self.y_lower

    @property
    def h(self):
        return self.length_x/self.rows_x

    @property
    def l(self):
        return self.length_y/self.rows_y


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
        return self.actual(self.x, self.y)


    @property
    def coefficients_Fx(self):
        """ Compute the Fourier coefficients for the partial first derivative
             with respect to x based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        row_indices = (i if i <= self.midpoint_y else (self.rows_y-i) for i in range(self.rows_y))
        coefficients = []
        for row_index in row_indices:
            row = []
            for column_index in range(self.rows_x):
                if not row_index and not column_index:
                    row.append(0)
                    continue
                k = row_index
                row.append(1j*2*math.pi*k/self.length_x)
            coefficients.append(
                np.fromiter(row, dtype=float)
            )
        return np.array(coefficients)

    @property
    def F_fft(self):
        return fft2(self.F)

    @property
    def Fx_complex(self):
        return ifft(self.F_fft*self.coefficients)

    @property
    def Fx_real(self):
        return np.real(self.Fx_complex)

    @property
    def Fx(self):
        return self.Fx_real


    @property
    def coefficients_Fy(self):
        """ Compute the Fourier coefficients for the partial first derivative with respect to
             y based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        row_indices = (i if i <= self.midpoint_y else (self.rows_y-i) for i in range(self.rows_y))
        coefficients = []
        for row_index in row_indices:
            row = []
            for column_index in range(self.rows_x):
                if not row_index and not column_index:
                    row.append(0)
                    continue
                elif column_index <= midpoint_y:
                    j = column_index
                else:
                    j = self.rows_x - column_index
                row.append(1j*2*math.pi*j/self.length_y)
            coefficients.append(
                np.fromiter(row, dtype=float)
            )
        return np.array(coefficients)

    @property
    def F_fft(self):
        return fft2(self.F)

    @property
    def Fx_complex(self):
        return ifft(self.F_fft*self.coefficients)

    @property
    def Fx_real(self):
        return np.real(self.Fx_complex)

    @property
    def Fx(self):
        return self.Fx_real





    def Fx(self):
        transformed = fft(self.F)
        coefficients = 1j*2*math.pi*self.indices_x/self.length_x
        return ifft(coefficients*transformed)

    def Fy(self):
        transformed = fft(self.F)
        

    def delta_p(self):
        # FIXME
        return self.Fx + self.Gy


    def mu(self):
        # FIXME
        pass

    @property
    def endpoint_factor(self):
        return -1 if self.edge_centered else 0

    @property
    def h(self):
        return (self.upper_bound - self.lower_bound) / (self.rows - self.endpoint_factor)

    @property
    def edge_centered(self):
        """ Our grid is edge-centered if either boundary condition is Dirichlet,
            and cell-centered otherwise
        """
        return False
0
    @property
    def mesh(self):
        """ Create an evenly spaced mesh of points representing the discretized
            domain of our problem
        """
        if self.edge_centered:
            return np.linspace(
                self.lower_bound + self.h,
                self.upper_bound,
                self.rows,
                endpoint=False
            )
        # If our grid is not edge centered, it's cell centered
        return np.linspace(
            self.lower_bound + self.h/2,
            self.upper_bound - self.h/2,
            self.rows,
            endpoint=True
        )

    @property
    def F(self):
        """ Given an ODE/PDE AU = F, return the source term F
        """
        F = np.apply_along_axis(self.f, 0, self.mesh)
        F = self.apply_boundary_conditions_f(F)
        return F

    @property
    def G(self):
        """ Given an ODE/PDE AU = F, return the source term F
        """
        G = np.apply_along_axis(self.g, 0, self.mesh)
        G = self.apply_boundary_conditions_f(G)
        return G

    @property
    def Fx(self):
        # FIXME
        transformed = fft(self.F)
        transformed[0] = 0

    @property
    def Gy(self):
        # FIXME
        transformed = fft(self.G)
        transformed[0] = 0

    def apply_boundary_conditions_f(self, F):
        if self.edge_centered:
            return self.build_edge_centered_f(F)
        return self.build_cell_centered_f(F)


    def build_edge_centered_f(self, F):
        if self.alpha.is_dirichlet:
            F[0] = F[0] - self.coef*self.alpha.value

        elif self.alpha.is_neumann:
            F[0] = F[0] + 2*self.alpha.value/(3*self.h)

        if self.beta.is_dirichlet:
            F[-1] = F[-1] - self.coef*self.beta.value

        elif self.beta.is_neumann:
            F[-1] = F[-1] - 2*self.beta.value/(3*self.h)

        return F


    def build_cell_centered_f(self, F):
        if self.alpha.is_neumann:
            F[0] = F[0] + self.alpha.value/self.h

        if self.beta.is_neumann:
            F[-1] = F[-1] - self.beta.value/self.h

        if self.alpha.is_periodic:
            pass

        return F


    @property
    def coef(self):
        """ A constant representing the value 1/h^2
        """
        return 1/self.h**2




