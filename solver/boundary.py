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

    def __init__(self, boundary_type, value=None):
        expected_boundary_types = [
            BCType.DIRICHLET,
            BCType.NEUMANN,
            BCType.PERIODIC
        ]

        if boundary_type not in expected_boundary_types:
            error_str = "The boundary_type must be one of {expected_boundary_types}, got {boundary_type} instead."
            raise Exception(error_str)

        if value is None and boundary_type != BCType.PERIODIC:
            error_str = "A value is required for non-periodic boundary conditions"
            raise Exception(error_str)

        # TODO: Throw an exception if bc is dirichlet or Neumann and value is Nonetes

        self.boundary_type = boundary_type
        self.value = value


    @property
    def is_dirichlet(self):
        return self.boundary_type == BCType.DIRICHLET


    @property
    def is_neumann(self):
        return self.boundary_type == BCType.NEUMANN


    @property
    def is_periodic(self):
        return self.boundary_type == BCType.PERIODIC



    def __repr__(self):
        if self.boundary_type != BCType.PERIODIC:
            return f"<BoundaryCondition {self.boundary_type} value:{self.value}>"
        return f"<BoundaryCondition {self.boundary_type}>"
