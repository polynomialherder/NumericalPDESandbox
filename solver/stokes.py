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

    def __init__(self, f, g, f_actual, g_actual, lower_bound, upper_bound, alpha, beta, rows):
        self.f = f
        self.g = g
        self.f_actual = f_actual
        self.g_actual = g_actual
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        self.beta = beta
        self.rows = rows

    def delta_p(self):
        # FIXME
        return self.Fx + self.Gy

    def delta_f(self):
        # FIXME
        pass

    def delta_g(self):
        # FIXME
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




