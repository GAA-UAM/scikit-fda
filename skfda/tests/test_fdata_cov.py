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
        100,
    ],
)
def n_samples(
    request: Any,
) -> Any:
    """Generate number of sample."""
    return request.param


@pytest.fixture(
    params=[
        100,
    ],
)
def n_features(
    request: Any,
) -> Any:
    """Generate number of features (points of evaluation)."""
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
        0.0,
    ],
)
def mean(
    request: Any,
) -> Any:
    """Generate a mean."""
    return request.param


@pytest.fixture(
    params=[
        0.1,
    ],
)
def noise(
    request: Any,
) -> Any:
    """Generate a noise value."""
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
    n_samples: int,
    n_features: int,
    interval: Tuple[float, float],
    mean: float,
    noise: float,
    covariance: CovarianceLike,
) -> FDataBasis:
    """Generate gaussian process data using a basis."""
    basis = basis_type(
        n_basis=n_basis,
        domain_range=interval,
    )
    return make_gaussian_process(
        n_samples=n_samples,
        n_features=n_features,
        start=interval[0],
        stop=interval[1],
        mean=mean,
        cov=covariance,
        noise=noise,
    ).to_basis(basis)


############
# TESTS
############

def test_fdatabasis_covariance(
    data_in_basis: FDataBasis,
    n_features: int,
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
    grid_points = np.linspace(domain_range[0], domain_range[1], n_features)
    data_in_grid = data_in_basis.to_grid(grid_points)

    s_grid, t_grid = grid_points, grid_points

    # Check that the covariance functions are equal
    np.testing.assert_allclose(
        data_in_basis.cov(s_grid, t_grid),
        data_in_grid.cov(s_grid, t_grid),
    )
