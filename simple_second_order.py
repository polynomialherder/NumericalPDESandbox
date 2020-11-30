"""
This module consists of abstractions for solving a simple linear second order differential equation of the form
f(x) = u''(x). The methods implemented are based on the discussion in Finite Difference Methods for Ordinary and Partial
Differential Equations by Randall Leveque, sections 2.1-2.10.
"""

import math

from functools import partial, cached_property

import matplotlib.pyplot as plt
import numpy as np


class SimpleSecondOrderODE:

    def __init__(self, f, h=0.1,
                 alpha=0, beta=50,
                 lower_bound=0, upper_bound=1,
                 actual=None
    ):
        """ Initialize a SimpleSecondOrderODE object. Given a source function
            f, transforms the problem into a linear equation AU = F where F = f(X)
            for a discretized domain X, and exposes methods for solving for U and
            analyzing errors given a known analytic solution (actual).
        """
        self.h = h
        self.f = f
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.actual = actual
        self.test_hs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    @property
    def coef(self):
        """ A constant representing the value 1/h^2
        """
        return 1/self.h**2


    @property
    def mesh(self):
        """ Create an evenly spaced mesh of points representing the discretized
            domain of our problem
        """
        return np.linspace(
            self.lower_bound + self.h,
            self.upper_bound,
            self.rows,
            endpoint=False
        )

    @property
    def F(self):
        """ Given an ODE/PDE AU = F, return the source term F
        """
        F = np.apply_along_axis(self.f, 0, self.mesh)
        F[0] = F[0] - self.alpha*self.coef
        F[-1] = F[-1] - self.beta*self.coef
        return F


    @property
    def rows(self):
        """ Compute the number of rows in the tridiagonal Toeplitz matrix A.
            We coerce the result to be an integer using ceil instead of int because
            floating point error occasionally results in strange values otherwise.
            The result must be an integer because we will be passing it to `np.full`
            which expects integer arguments
        """
        return math.ceil((self.upper_bound - self.lower_bound) / self.h - 1)


    @property
    def A(self):
        """ Determine the tridiagonal Toeplitz matrix that determines our finite
            difference approximation to the second derivative by summing three diagonal
            matrices
        """
        A_ = np.diag(np.full(self.rows -1, 1), k=-1) + \
            np.diag(np.full(self.rows, -2), k=0) + \
            np.diag(np.full(self.rows - 1, 1), k=1)
        return self.coef * A_


    def apply_actual(self):
        """ Apply the known solution to our mesh values
        """
        return np.apply_along_axis(self.actual, 0, self.mesh)


    def residuals(self):
        """ The signed difference between our approximation and the known solution
            Synonymous with the gte (global truncation error) method and likely more
            efficient
        """
        return self.solution - self.apply_actual()


    @property
    def solution(self):
        return self.solve()


    def solve(self):
        """ Solve an ODE/PDE in the form AU = F for U
        """
        return np.linalg.solve(self.A, self.F)


    def lte(self):
        """ A method returning the local truncation error as defined by Leveque,
            namely A*U_hat - F where A is our differential approximation operator,
            U_hat is a vector of known solutions, and F is our source function.
        """
        u_hat = self.apply_actual()
        AU_hat = self.A @ u_hat
        return AU_hat - self.F


    def gte(self):
        """ The global truncation error as defined by Leveque, namely
            -(A^{-1})T where T is the local truncation error. Synonymous
            with the residuals method (which is likely more performant).
        """
        return -(np.linalg.inv(self.A)) @ self.lte()


    def A_inv(self):
        """ Invert our differential approximation operator.
        """
        return np.linalg.inv(self.A)


    def check_stability(self):
        """ Implements a check of a handful of 2-norms corresponding to reasonable values
            of h as a quick check that they satisfy the following definition from Leveque:

            Given a finite difference method for a linear BVP, the method is stable
            if (A)^{-1} exists for all h sufficiently small, and if there is a constant C,
            independent of h, such that ||A^{-1}|| <= C for all h < h_0
        """
        norms = []
        prev = self.h
        for h in self.test_hs:
            self.h = h
            two_norm = np.linalg.norm(self.A_inv, 2)
            norms.append(two_norm)
        self.h = prev
        return norms


    def plot_approximation(self):
        plt.plot(self.mesh, self.apply_actual(), label="Analytic Solution")
        plt.plot(self.mesh, self.solution, label="FD Approximation")
        plt.title("Approximat")
        plt.xlabel("x")
        plt.ylabel("U(x)")
        plt.legend()
        plt.show()

    def plot_h_vs_error(self):
        errors = []
        for h in self.test_hs:
            eqn.h = h
            errors.append(_2norm(eqn.h, eqn.gte()))
        errors = np.log(errors)
        hs = np.log(self.test_hs)
        plt.plot(hs, errors)
        plt.xlabel("h")
        plt.ylabel("Global Truncation Error")
        plt.show()


def _2norm(h, vector):
    """ Implements the 2-norm as described by Leveque on page 17
    """
    return math.sqrt(h * sum(abs(vector)**2))


if __name__ == '__main__':
    # Function representing choice of f(x) (as in the steady-state PDE u''(x) = f(x))
    f = lambda x: (np.cos(x)) ** 2
    # Function representing an analytic solution for u of the equation u'' = f(x)
    actual = lambda x: (1/80)*(20*x**2 + x*(191 + np.cos(20)) - 10*np.cos(2*x) + 90)
    eqn = SimpleSecondOrderODE(f, h=0.001, lower_bound=-10, upper_bound=10, actual=actual, alpha=0, beta=50)
