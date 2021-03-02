"""
This module consists of abstractions for solving a simple linear second order differential equation of the form
f(x) = u''(x). The methods implemented are based on the discussion in Finite Difference Methods for Ordinary and Partial
Differential Equations by Randall Leveque, sections 2.1-2.10.
"""
import math

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from solver.boundary import BCType, BoundaryCondition


class PoissonSolver:

    def __init__(self, f, h=0.1,
                 alpha=0, beta=0,
                 lower_bound=0, upper_bound=1,
                 actual=None, dense=False
    ):
        """ Initialize a SimpleSecondOrderODE object. Given a source function
            f, transforms the problem into a linear equation AU = F where F = f(X)
            for a discretized domain X, and exposes methods for solving for U and
            analyzing errors given a known analytic solution (actual).

            Supports Neumann and Dirichlet boundary conditions
        """
        self.h = h
        self.f = f
        self.dense = dense
        self.alpha = alpha
        self.beta = beta
        self.set_boundary_conditions()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.actual = actual
        self.test_hs = [1/1000, 1/500, 1/320, 1/250, 1/125, 1/100, 1/50, 1/30, 1/25, 1/20, 1/10, 1/5]

    def set_boundary_conditions(self):
        # These should be dynamic properties with getter/setter methods
        # that automatically call normalize_boundary_condition
        self.alpha = self.normalize_boundary_condition(self.alpha)
        if self.alpha == BCType.PERIODIC:
            self.beta = self.alpha
        else:
            self.beta = self.normalize_boundary_condition(self.beta)


    @staticmethod
    def normalize_boundary_condition(boundary_value):
        if isinstance(boundary_value, BoundaryCondition):
            return boundary_value
        elif isinstance(boundary_value, (complex, float, int)):
            return BoundaryCondition(BCType.DIRICHLET, boundary_value)
        raise Exception("boundary_value must be either a BoundaryCondition object or a numeric type")


    @property
    def coef(self):
        """ A constant representing the value 1/h^2
        """
        return 1/self.h**2


    @property
    def edge_centered(self):
        """ Our grid is edge-centered if either boundary condition is Dirichlet,
            and cell-centered otherwise
        """
        return BCType.DIRICHLET in (self.alpha.boundary_type, self.beta.boundary_type)


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
    def endpoint_factor(self):
        return -1 if self.edge_centered else 0


    @property
    def rows(self):
        """ Compute the number of rows in the (mostly) tridiagonal Toeplitz matrix A
            that represents our differential operator

            Note that we coerce the result to be an integer using ceil instead of int because
            floating point error occasionally results in strange values otherwise.
            The result must be an integer because we will be passing it to `np.full`
            which expects integer arguments
        """
        return math.ceil((self.upper_bound - self.lower_bound) / self.h) + self.endpoint_factor

    def apply_boundary_conditions_f(self, F):
        if self.alpha.is_dirichlet:
            F[0] = F[0] - self.coef*self.alpha.value

        elif self.alpha.is_neumann:
            F[0] = F[0]/2 + 1/self.h * self.alpha.value

        if self.beta.is_dirichlet:
            F[-1] = F[-1] - self.coef*self.beta.value

        elif self.beta.is_neumann:
            F[-1] = F[-1] - 2/self.h * self.beta.value

        return F


    def apply_boundary_conditions_A(self, A):
        if self.alpha.is_dirichlet:
            A[0, 0] = -2
            A[0, 1] = 1

        elif self.alpha.is_neumann:
            A[0, 0] = -1
            A[0, 1] = 1

        if self.beta.is_dirichlet:
            A[-1, -1] = -2
            A[-1, -2] = 1

        if self.beta.is_neumann:
            A[-1, -2] = 1
            A[-1, -1] = -1

        if self.alpha.is_periodic:
            A[0, -1] = 1
            A[-1, 0] = 1

        return A


    @property
    def A(self):
        """ Determine the tridiagonal Toeplitz matrix that determines our finite
            difference approximation to the second derivative by summing three diagonal
            matrices
        """
        if self.dense:
            A_ = self.build_dense_base_matrix()
        else:
            A_ = self.build_sparse_base_matrix()

        A_ = self.apply_boundary_conditions_A(A_)

        if not self.dense:
            A_ = csr_matrix(A_)

        return self.coef * A_


    def build_dense_base_matrix(self):
        """ Build a dense base matrix of size rows-2 x rows-2
        """
        A = np.diag(np.full(self.rows-3, 1), k=-1) + \
             np.diag(np.full(self.rows-2, -2), k=0) + \
             np.diag(np.full(self.rows-3, 1), k=1)
        left_column = np.zeros(A.shape[0])
        left_column[0] = 1
        right_column = np.zeros(A.shape[0])
        right_column[-1] = 1
        A = np.column_stack((left_column, A, right_column))
        # After the np.column_stack call A looks like
        #
        # | 1 -2  1 0 ...  0 0 0 |
        # | 0  1 -2 1 ...  0 0 0 |
        # |    ...    ... ... ...|
        # | 0  0  0 0 ... 1 -2 1 |
        #
        # Create an array of zeros for the first and last rows
        boundary = np.zeros(A.shape[1])
        A = np.vstack(
            (
                boundary,
                A,
                boundary
            )
        )
        # Now we have:
        #
        # | 0  0  0 0 ... 0  0 0 |
        # | 1 -2  1 0 ...  0 0 0 |
        # | 0  1 -2 1 ...  0 0 0 |
        # |    ...    ... ... ...|
        # | 0  0  0 0 ... 1 -2 1 |
        # | 0  0  0 0 ... 0  0 0 |
        #
        # so that applying the boundary conditions to A is a matter of
        # modifying the first and last rows
        return A



    def build_sparse_base_matrix(self):
        """ Build a sparse list-of-lists base matrix. This matrix
            should be converted to a more efficient sparse representation
            for algebraic operations (such as CSR) prior to solving
        """
        A_ = lil_matrix(
            (self.rows, self.rows)
        )
        A_.setdiag(1, 1)
        A_.setdiag(-2, 0)
        A_.setdiag(1, -1)
        # We'd like to modify the boundaries later, so set them to zero
        # for now
        A_[0, 0] = 0
        A_[0, 1] = 0
        A_[0, 2] = 0
        A_[-1, -1] = 0
        A_[-1, -2] = 0
        A_[-1, -3] = 0
        return A_


    def apply_actual(self):
        """ Apply the known solution to our mesh values
        """
        return np.apply_along_axis(self.actual, 0, self.mesh)


    def residuals(self):
        """ The signed difference between our approximation and the known solution
            Synonymous with the gte (global truncation error) method and likely more
            efficient
        """
        return self.A @ self.solution - self.F


    @property
    def solution(self):
        return self.solve()


    def solve(self):
        """ Solve an ODE/PDE in the form AU = F for U
        """
        if self.dense:
            return np.linalg.solve(self.A, self.F)
        return spsolve(self.A, self.F)


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
            with SimpleSecondOrderODE.residuals method (which is likely more performant).
        """
        return self.solution - self.apply_actual()


    @property
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


    def check_consistency(self):
        _2norms = []
        for h in self.test_hs:
            self.h = h
            lte = self.lte()
            _2norms.append(_2norm(self.h, lte))
        return _2norms


    def plot_approximation(self):
        plt.plot(self.mesh, self.apply_actual(), label="Analytic Solution")
        plt.plot(self.mesh, self.A @ self.solution, label="FD Approximation")
        plt.title("Approximate")
        plt.xlabel("x")
        plt.ylabel("U(x)")
        plt.legend()
        plt.show()

    def plot_h_vs_error(self, subtitle=""):
        errors = []
        for h in self.test_hs:
            self.h = h
            errors.append(_2norm(self.h, self.gte()))
        errors = (abs(np.array(errors)))
        hs = (self.test_hs)
        plt.loglog(hs, errors)
        plt.xlabel("h")
        plt.ylabel("Global Error")
        plt.title(f"Error vs h{'' if not subtitle else ': ' + subtitle}")
        plt.show()


def _2norm(h, vector):
    """ Implements the 2-norm as described by Leveque on page 17
    """
    return math.sqrt(h * sum(abs(vector)**2))


def rmse(actual, approximations):
    return (sum((approximations - actual) ** 2) / len(actual)) ** (1/2)


def l1_norm(actual, reference):
    return (1/len(actual)) * sum(abs(actual - reference))


def infinity_norm(actual, reference):
    return max(abs(actual - reference))
