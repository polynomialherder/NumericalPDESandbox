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
from scipy.sparse.linalg import spsolve, lsqr

from solver.boundary import BCType, BoundaryCondition


class PoissonSolver:

    def __init__(self, f, rows=10,
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
        self.f = f
        self.rows = rows
        self.dense = dense
        self.alpha = alpha
        self.beta = beta
        self.set_boundary_conditions()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.actual = actual
        self.suppress_warnings = False
        self.least_squares_solution = False
        self.test_rows = [5120, 2560, 1280, 640, 320, 160, 80, 40, 20]

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
        return False
        #return BCType.DIRICHLET in (self.alpha.boundary_type, self.beta.boundary_type)


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
    def has_solution(self):
        if self.edge_centered:
            return True

        return (abs(sum(self.F))) < 10e-10


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
    def h(self):
        return (self.upper_bound - self.lower_bound) / (self.rows - self.endpoint_factor)


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

        return F


    def build_edge_centered_A(self, A):
        if self.alpha.is_dirichlet:
            A[0, 0] = -2
            A[0, 1] = 1

        elif self.alpha.is_neumann:
            A[0, 0] = -2/3
            A[0, 1] = 2/3

        if self.beta.is_dirichlet:
            A[-1, -1] = -2
            A[-1, -2] = 1

        if self.beta.is_neumann:
            A[-1, -2] = 2/3
            A[-1, -1] = -2/3

        if self.alpha.is_periodic or self.beta.is_periodic:
            A[0, 0] = -2
            A[0, 1] = 1
            A[0, -1] = 1
            A[-1, 0] = 1
            A[-1, -2] = 1
            A[-1, -1] = -2
        return A


    def build_cell_centered_A(self, A):
        if self.alpha.is_neumann:
            A[0, 0] = -1
            A[0, 1] = 1
        if self.beta.is_neumann:
            A[-1, -2] = 1
            A[-1, -1] = -1
        return A


    def apply_boundary_conditions_A(self, A):
        if self.edge_centered:
            return self.build_edge_centered_A(A)
        return self.build_cell_centered_A(A)


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
    def is_singular(self):
        self.dense = True
        singular = not (np.linalg.matrix_rank(self.A) == self.A.shape[0])
        self.dense = False
        return singular


    @property
    def solution(self):
        solution = self.solve()
        integral = 0
        if self.least_squares_solution:
            integral = np.cumsum(solution)[-1] / self.rows
        return solution - integral

    @property
    def U(self):
        return self.solution


    def print_singular_warning(self):
        if not self.suppress_warnings:
            warning = f"Warning: A is singular and "
            warning += "A @ U = F has infinitely many solutions, "
            warning += "returning a least-squares approximation"
            print(warning)


    def solve_dense(self):
        try:
            solution = np.linalg.solve(self.A, self.F)
            self.least_squares_solution = False
            return solution
        except np.linalg.LinAlgError:
            if self.has_solution:
                self.print_singular_warning()
                self.least_squares_solution = True
                return np.linalg.lstsq(self.A, self.F)[0]
            raise Exception(f"The linear operator A is singular and has no solutions")


    def solve_sparse(self):
        # FIXME
        # We check if the matrix is singular by calling spsolve, and checking if
        # the minimum value of the array is nan. If it is, then the array was all
        # nan, and so the matrix was singular, and we proceed with a last squares
        # approximation instead. Otherwise, the matrix is nonsingular, in which case
        # we just return the solution given by spsolve.
        #
        # This heuristic for checking if a matrix is singular is problematic as
        # in certain cases spsolve will return a value even for singular matrices
        # that differs substantially from what lsqr returns.
        #
        # Note that this isn't (alone) causing accuracy problems since for no test
        # rowsize used in self.plot_h_vs_error is spsolve returning something non-nan
        # (and therefore believing the matrix to be nonsingular).
        maybe_solution = spsolve(self.A, self.F)
        singular = np.isnan(min(maybe_solution))
        if not singular:
            print(f"Not singular")
            self.least_squares_solution = False
            return maybe_solution

        if self.has_solution:
            self.print_singular_warning()
            self.least_squares_solution = True
            return lsqr(self.A, self.F)[0]
        raise Exception(f"The linear operator A is singular and AU = F has no solutions")


    def solve(self):
        """ Solve an ODE/PDE in the form AU = F for U
        """
        if self.dense:
            return self.solve_dense()

        return self.solve_sparse()


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
            -(A^{-1})T where T is the local truncation error. 
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
        for row in self.test_rows:
            self.rows = row
            two_norm = np.linalg.norm(self.A_inv, 2)
            norms.append(two_norm)
        self.h = prev
        return norms


    def check_consistency(self):
        _2norms = []
        for row in self.test_rows:
            self.rows = row
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
        hs = []
        for idx, row in enumerate(self.test_rows):
            self.rows = row
            hs.append(self.h)
            errors.append(_2norm(self.h, self.gte()))
            # On the first iteration, we set the suppress_warnings
            # flag so that only one message is printed
            if not idx:
                self.suppress_warnings = True
        self.suppress_warnings = False
        errors = abs(np.array(errors))
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
