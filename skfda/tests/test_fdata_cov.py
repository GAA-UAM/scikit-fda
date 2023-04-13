"""Test the covariance method of FData."""
from typing import Any, Callable, Tuple

import numpy as np
import pytest

from skfda.datasets import make_gaussian_process
from skfda.misc.covariances import CovarianceLike, Gaussian
from skfda.representation import FDataBasis
from skfda.representation.basis import Basis, BSplineBasis, FourierBasis

############
# FIXTURES
############

N_SAMPLES = 100
N_FEATURES = 100


@pytest.fixture(
    params=[
        FourierBasis,
        BSplineBasis,
    ],
    ids=[
        "FourierBasis",
        "BSplineBasis",
    ],
)
def basis_type(
    request: Any,
) -> Any:
    """Fixture for classes to test."""
    return request.param


@pytest.fixture(
    params=[
        5,
        30,
    ],
)
def n_basis(
    request: Any,
) -> Any:
    """Generate a basis."""
    return request.param


@pytest.fixture(
    params=[
        (0.0, 1.0),
        (-2.0, 2.0),
    ],
)
def interval(
    request: Any,
) -> Any:
    """Generate an interval."""
    return request.param


@pytest.fixture(
    params=[
        Gaussian(
            variance=1,
            length_scale=1,
        ),
        Gaussian(
            variance=0.1,
            length_scale=1,
        ),
        Gaussian(
            variance=3,
            length_scale=0.5,
        ),
    ],
)
def covariance(
    request: Any,
) -> Any:
    """Generate a covariance kernel."""
    return request.param


@pytest.fixture
def data_in_basis(
    basis_type: Callable[..., Basis],
    n_basis: int,
    interval: Tuple[float, float],
    covariance: CovarianceLike,
) -> FDataBasis:
    """Generate gaussian process data using a basis."""
    basis = basis_type(
        n_basis=n_basis,
        domain_range=interval,
    )
    return make_gaussian_process(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        start=interval[0],
        stop=interval[1],
        cov=covariance,
    ).to_basis(basis)


############
# TESTS
############

def test_overload_cov() -> None:
    """Test that the overloading of the cov method is consistent."""
    # Generate any sample data
    data = make_gaussian_process(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        start=0,
        stop=1,
    )
    grid_points = np.linspace(0, 1, N_FEATURES)
    # Test for FDataGrid
    np.testing.assert_equal(
        data.cov()(grid_points, grid_points),
        data.cov(grid_points, grid_points),
    )
    # Test for FDataBasis
    basis = FourierBasis(n_basis=5, domain_range=(0, 1))
    data_in_basis = data.to_basis(basis)
    np.testing.assert_equal(
        data_in_basis.cov()(grid_points, grid_points),
        data_in_basis.cov(grid_points, grid_points),
    )


def test_fdatabasis_covariance(
    data_in_basis: FDataBasis,
) -> None:
    """Test the covariance method of FDataBasis.

    The resulting covariance function is defined on the tensor basis that is
    the tensor product of the basis of the original functions with itself.

    This test checks that the covariance function calculated for FDataBasis
    objects is equal to the covariance function calculated for the grid
    representation of the same data.
    """
    # Select grid points before converting to grid
    domain_range = data_in_basis.domain_range[0]
    grid_points = np.linspace(domain_range[0], domain_range[1], N_FEATURES)
    data_in_grid = data_in_basis.to_grid(grid_points)
    # Check that the covariance functions are equal
    np.testing.assert_allclose(
        data_in_basis.cov(grid_points, grid_points),
        data_in_grid.cov(grid_points, grid_points),
    )
