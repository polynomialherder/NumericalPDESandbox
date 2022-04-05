"""
This module consists of abstractions for solving Stokes equation in 2D
"""
import math

from enum import Enum
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy.linalg import norm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, lsqr, svds

from solver.boundary import BCType, BoundaryCondition

class StokesSolver:
    def __init__(
        self,
        xv,
        yv,
        F,
        G,
        mu=1
    ):
        self.x = xv
        self.y = yv
        self.h = (self.x[0][1] - self.x[0][0])/self.x[0].size
        self.l = (self.y[1][0] - self.y[0][0])/len(self.y)
        self.F = F
        self.G = G
        self.mu = mu
        self.length_x = self.x[0][-1] - self.x[0][0]
        self.length_y = self.y[-1][0] - self.y[0][0]
        self.rows_x = self.x[0].size
        self.rows_y = len(self.y)


    @property
    def midpoint_x(self):
        return self.rows_x // 2

    @property
    def midpoint_y(self):
        return self.rows_y // 2

    @property
    def fourier_indices_x(self):
        return np.array(
            [
                i if i <= self.midpoint_x else -(self.rows_x - i)
                for i in range(self.rows_x)
            ]
        )

    @property
    def fourier_indices_y(self):
        return np.array(
            [
                i if i <= self.midpoint_y else -(self.rows_y - i)
                for i in range(self.rows_y)
            ]
        )

    @property
    def fourier_k(self):
        k = []
        for index in self.fourier_indices_x:
            k.append([index for _ in range(self.rows_x)])
        return np.array(k)

    @property
    def fourier_j(self):
        return np.array([self.fourier_indices_y for _ in range(self.rows_y)])


    @property
    def _x(self):
        return np.linspace(
            self.x_lower + self.h / 2,
            self.x_upper - self.h / 2,
            self.rows_x,
            endpoint=True,
        )

    @property
    def _y(self):
        return np.linspace(
            self.y_lower + self.l / 2,
            self.y_upper - self.l / 2,
            self.rows_y,
            endpoint=True,
        )

    @property
    def v_actual(self):
        return self.v_actual_(self.x, self.y)

    @property
    def u_actual(self):
        return self.u_actual_(self.x, self.y)

    @property
    def p_actual(self):
        return self.p_actual_(self.x, self.y)

    @property
    def coefficients_Dx(self):
        """Compute the Fourier coefficients for the partial first derivative
        with respect to x based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        eig_factor = 1j * 2 * math.pi / self.length_x
        return eig_factor * self.fourier_k

    @property
    def F_fourier(self):
        return fft2(self.F)

    @property
    def Fx_fourier(self):
        return self.F_fourier * self.coefficients_Dx

    @property
    def coefficients_Dy(self):
        """Compute the Fourier coefficients for the partial first derivative with respect to
        y based on the scipy fft implementation
        """
        # row_indices is a generator with k-values in the right places per Python's fft
        # transformation implementation. For example, given rows = 9, row_indices will
        # be a generator containing following values:
        #   [0, 1, 2, 3, 4, -4, -3, -2, -1]
        eig_factor = 1j * 2 * math.pi / self.length_y
        return eig_factor * self.fourier_j

    @property
    def Gy_fourier(self):
        return self.G_fourier * self.coefficients_Dy

    def poisson_solve_fft(self, matrix):
        denominator_term_x = lambda n: (2 * math.pi * n / self.length_x) ** 2
        denominator_term_y = lambda n: (2 * math.pi * n / self.length_y) ** 2
        coefficients = denominator_term_x(self.fourier_k) + denominator_term_y(
            self.fourier_j
        )
        coefficients[0, 0] = 1
        matcoef = -matrix / coefficients
        matcoef[0, 0] = 0
        return matcoef

    @property
    def p_fourier(self):
        return self.poisson_solve_fft(-self.Fx_fourier + -self.Gy_fourier)

    @property
    def px_fourier(self):
        return self.p_fourier * self.coefficients_Dx

    @property
    def py_fourier(self):
        return self.p_fourier * self.coefficients_Dy

    @property
    def p_ifft(self):
        return ifft2(self.p_fourier)

    @property
    def p(self):
        return np.real(self.p_ifft)

    @property
    def px_ifft(self):
        return ifft2(self.px_fourier)

    @property
    def px(self):
        return np.real(self.px_ifft)

    @property
    def py_ifft(self):
        return ifft2(self.py_fourier)

    @property
    def py(self):
        return np.real(self.py_ifft)

    @property
    def u_fourier(self):
        grad_u = (1 / self.mu) * (self.px_fourier + self.F_fourier)
        return self.poisson_solve_fft(grad_u)

    @property
    def u_ifft(self):
        return ifft2(self.u_fourier)

    @property
    def u(self):
        return np.real(self.u_ifft)

    @property
    def v_fourier(self):
        grad_v = (1 / self.mu) * (self.py_fourier + self.G_fourier)
        return self.poisson_solve_fft(grad_v)

    @property
    def v_ifft(self):
        return ifft2(self.v_fourier)

    @property
    def v(self):
        return np.real(self.v_ifft)

    @property
    def G_fourier(self):
        return fft2(self.G)

    @property
    def H_complex(self):
        return ifft2(self.Fx_fourier + self.Gy_fourier)

    @property
    def H_real(self):
        return np.real(self.H_complex)

    @property
    def H(self):
        return self.H_real

    def error(self, v1, v2, p=2):
        return norm(v1 - v2, p)

    def error_v(self, p=2):
        return self.error(self.v_actual, self.v, p=p)

    def error_u(self, p=2):
        return self.error(self.u_actual, self.u, p=p)

    def error_p(self, p=2):
        return self.error(self.p_actual, self.p, p=p)
