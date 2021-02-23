import pytest

import numpy as np
from numpy import testing

from solver.boundary import BCType, BoundaryCondition
from solver.second_order import PoissonSolver

@pytest.fixture
def eqn_with_dirichlet_bc():
    eqn = PoissonSolver(
        f=lambda x: x,
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        actual=lambda x: x,
        alpha=0,
        beta=1
    )
    return eqn


@pytest.fixture
def eqn_with_neumann_bc():
    bc = BoundaryCondition(
        BCType.NEUMANN,
        1
    )
    eqn = PoissonSolver(
        f=lambda x: x,
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        actual=lambda x: x,
        alpha=bc,
        beta=bc
    )
    return eqn


@pytest.fixture
def eqn_with_mixed_bc_types():
    bc1 = BoundaryCondition(
        BCType.NEUMANN,
        1
    )
    bc2 = BoundaryCondition(
        BCType.DIRICHLET,
        1
    )
    return PoissonSolver(
        f=lambda x: x,
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        actual=lambda x: x,
        alpha=bc1,
        beta=bc2
    )


@pytest.fixture
def eqn_with_periodic_bc():
    bc = BoundaryCondition(
        BCType.PERIODIC,
        1
    )
    return PoissonSolver(
        f=lambda x: x,
        h=0.1,
        lower_bound=0,
        upper_bound=1,
        alpha=bc,
        beta=bc
    )


def test_dirichlet_bc_induces_edge_centered_grid(eqn_with_dirichlet_bc):
    eqn = eqn_with_dirichlet_bc

    assert eqn.edge_centered

    eqn.h = 0.1
    mesh = np.array([
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ])
    testing.assert_array_almost_equal(eqn.mesh, mesh)


def test_neumann_bc_induces_cell_centered_grid(eqn_with_neumann_bc):
    eqn = eqn_with_neumann_bc

    assert not eqn.edge_centered

    eqn.h = 0.1
    mesh = np.array([
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95
    ])
    testing.assert_array_almost_equal(eqn.mesh, mesh)


def test_periodic_bc_induces_cell_centered_grid(eqn_with_periodic_bc):
    eqn = eqn_with_periodic_bc

    assert not eqn.edge_centered

    eqn.h = 0.1
    mesh = np.array([
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95
    ])
    testing.assert_array_almost_equal(eqn.mesh, mesh)


def test_mixed_bcs_induce_edge_centered_grid(eqn_with_mixed_bc_types):
    eqn = eqn_with_mixed_bc_types

    assert eqn.edge_centered

    eqn.h = 0.1
    mesh = np.array([
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ])
    testing.assert_array_almost_equal(eqn.mesh, mesh)


