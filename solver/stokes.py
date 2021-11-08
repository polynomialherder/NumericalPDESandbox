"""
This module consists of abstractions for solving a simple linear second order differential equation of the form
f(x) = u''(x). The methods implemented are based on the discussion in Finite Difference Methods for Ordinary and Partial
Differential Equations by Randall Leveque, sections 2.1-2.10.
"""
import math

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, lsqr, svds

from solver.boundary import BCType, BoundaryCondition

MACHINE_EPSILON = 2e-16

class StokesSolver:

    def __init__(self, f, g, f_actual, g_actual, rows_x, rows_y, x_lower, x_upper, y_lower, y_upper):
        self.f = f
        self.g = g
        self.rows_x = rows_x
        self.rows_y = rows_y
        self.f_actual = f_actual
        self.g_actual = g_actual
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper


    @property
    def midpoint_x(self):
        return self.rows_x // 2

    @property
    def midpoint_y(self):
        return self.rows_y // 2

    @property
    def fourier_indices_x(self):
        return np.array([i if i <= self.midpoint_x else -(self.rows_x - i) for i in range(self.rows_x)])

    @property
    def fourier_indices_y(self):
        return np.array([i if i <= self.midpoint_y else -(self.rows_y - i) for i in range(self.rows_y)])

    @property
    def fourier_k(self):
        k = []
        for index in self.fourier_indices_x:
            k.append(
                [index for _ in range(self.rows_x)]
            )
        return np.array(k)

    @property
    def fourier_j(self):
        return np.array([self.fourier_indices_y for _ in range(self.rows_y)])

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
        return self.f(self.x, self.y)

    @property
    def coefficients_Fx(self):
        """ Compute the Fourier coefficients for the partial first derivative
             with respect to x based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        eig_factor = 1j*2*math.pi/self.length_x
        return eig_factor*self.fourier_indices_x

    @property
    def F_fft(self):
        return fft2(self.F)

    @property
    def Fx_fourier(self):
        return self.F_fft*self.coefficients_Fx


    @property
    def G(self):
        self.x, self.y = self.meshgrid
        return self.g(self.x, self.y)


    @property
    def coefficients_Gy(self):
        """ Compute the Fourier coefficients for the partial first derivative with respect to
             y based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        eig_factor = 1j*2*math.pi/self.length_y
        return eig_factor*self.fourier_indices_y

    @property
    def G_fft(self):
        return fft2(self.G)

    @property
    def Gy_fourier(self):
        return self.G_fft*self.coefficients_Gy


    @property
    def H_complex(self):
        return ifft2(self.Fx_fourier + self.Gy_fourier)

    @property
    def H_real(self):
        return np.real(self.H_complex)

    @property
    def H(self):
        return self.H_real


    @property
    def p(self):
        pass



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




