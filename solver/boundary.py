""" This module consists of abstractions for specifying boundary conditions.
"""
from enum import Enum


class BCType(Enum):
    """ An enum value representing either a Dirichlet boundary condition
        or a Neumann boundary condition
    """

    DIRICHLET = 1
    NEUMANN = 2
    PERIODIC = 3


class BoundaryCondition:
    """ A BoundaryCondition object contains the information necessary to apply
        boundary conditions to the solution of the PDE -- namely, the boundary type,
        which must be one of BCType.DIRICHLET or BCType.NEUMANN, and the value of
        u(gamma) or u'(gamma) at the boundary where gamma is the boundary point and
        u is the solution of the PDE.
    """

    def __init__(self, boundary_type, value, nth_derivative=0):
        expected_boundary_types = [
            BCType.DIRICHLET,
            BCType.NEUMANN,
            BCType.PERIODIC
        ]

        if boundary_type not in expected_boundary_types:
            error_str = "The boundary_type must be one of {expected_boundary_types}, got {boundary_type} instead."
            raise Exception(error_str)

        self.boundary_type = boundary_type
        self.value = value

        if self.boundary_type == BCType.PERIODIC and nth_derivative > 1:
            raise Exception(f"Periodic boundary conditions may only be first or second derivatives")
        self._nth_derivative = nth_derivative


    @property
    def nth_derivative(self):
        if self.boundary_type == BCType.NEUMANN:
            self._nth_derivative = 1
        elif self.boundary_type == BCType.DIRICHLET:
            self._nth_derivative = 0
        return self._nth_derivative


    def __repr__(self):
        return f"<BoundaryCondition {self.boundary_type} value:{self.value}>"
