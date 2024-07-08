"""Tests for Covariance module."""
from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid

import skfda.misc.covariances as cov
from skfda import FDataBasis, FDataGrid
from skfda.datasets import fetch_weather
from skfda.representation.basis import MonomialBasis


def _test_compare_sklearn(
    multivariate_data: Any,
    cov: cov.Covariance,
) -> None:
    cov_sklearn = cov.to_sklearn()
    cov_matrix = cov(multivariate_data)
    cov_sklearn_matrix = cov_sklearn(multivariate_data)

    np.testing.assert_array_almost_equal(cov_matrix, cov_sklearn_matrix)

###############################################################################
# Example datasets for which to calculate the evaluation of the kernel by hand
# to compare it against the results yielded by the implementation.
###############################################################################


basis = MonomialBasis(n_basis=2, domain_range=(-2, 2))

fd = [
    FDataBasis(basis=basis, coefficients=[[1, 0], [1, 2]]),
    FDataBasis(basis=basis, coefficients=[[0, 1]]),
]

##############################################################################
# Fixtures
##############################################################################


@pytest.fixture
def fetch_weather_subset() -> FDataGrid:
    """Fixture for loading the canadian weather dataset example."""
    fd, _ = fetch_weather(return_X_y=True)
    return fd[:20]


@pytest.fixture(
    params=[
        cov.Linear(),
        cov.Polynomial(),
        cov.Gaussian(),
        cov.Exponential(),
        cov.Matern(),
    ],
)
def covariances_fixture(request: Any) -> Any:
    """Fixture for getting a covariance kernel function."""
    return request.param


@pytest.fixture(
    params=[
        cov.Brownian(),
        cov.WhiteNoise(),
    ],
)
def covariances_raise_fixture(request: Any) -> Any:
    """Fixture for getting a covariance kernel that raises a ValueError."""
    return request.param


@pytest.fixture(
    params=[
        [
            cov.Linear(variance=1 / 2, intercept=3),
            np.array([[3 / 2], [3 / 2 + 32 / 6]]),
        ],
        [
            cov.Polynomial(variance=1 / 3, slope=2, intercept=1, degree=2),
            np.array([[1 / 3], [67**2 / 3**3]]),
        ],
        [
            cov.Gaussian(variance=3, length_scale=2),
            np.array([[3 * np.exp(-7 / 6)], [3 * np.exp(-7 / 6)]]),
        ],
        [
            cov.Exponential(variance=4, length_scale=5),
            np.array([
                [4 * np.exp(-np.sqrt(28 / 3) / 5)],
                [4 * np.exp(-np.sqrt(28 / 3) / 5)],
            ]),
        ],
        [
            cov.Matern(variance=2, length_scale=3, nu=2),
            np.array([
                [(2 / 3) ** 2 * (28 / 3) * 0.239775899566],
                [(2 / 3) ** 2 * (28 / 3) * 0.239775899566],
            ]),
        ],
    ],
)
def precalc_example_data(
    request: Any,
) -> list[FDataBasis, FDataBasis, cov.Covariance, np.array]:
    """Fixture for getting fdatabasis objects.

    The dataset is used to test manual calculations of the covariance functions
    against the implementation.
    """
    # First fd, Second fd, kernel used, result
    return *fd, *request.param


@pytest.fixture
def multivariate_data() -> np.array:
    """Fixture for getting multivariate data."""
    return np.linspace(-1, 1, 1000)[:, np.newaxis]


@pytest.fixture(
    params=[
        [
            cov.Linear,
            {
                "variance": [1, 2],
                "intercept": [3, 4],
            },
        ],
        [
            cov.Polynomial,
            {
                "variance": [2],
                "intercept": [0, 2],
                "slope": [1, 2],
                "degree": [1, 2, 3],
            },
        ],
        [
            cov.Exponential,
            {
                "variance": [1, 2],
                "length_scale": [0.5, 1, 2],
            },
        ],
        [
            cov.Gaussian,
            {
                "variance": [1, 2],
                "length_scale": [0.5, 1, 2],
            },
        ],
        [
            cov.Matern,
            {
                "variance": [2],
                "length_scale": [0.5],
                "nu": [0.5, 1, 1.5, 2.5, 3.5, np.inf],
            },
        ],
    ],
)
def covariance_and_params(request: Any) -> Any:
    """Fixture to load the covariance functions."""
    return request.param


##############################################################################
# Tests
##############################################################################


def test_covariances(
    fetch_weather_subset: FDataGrid,
    covariances_fixture: cov.Covariance,
) -> None:
    """Check that parameter conversion is done correctly."""
    fd = fetch_weather_subset
    cov_kernel = covariances_fixture

    # Also test that it does not fail
    res1 = cov_kernel(fd, fd)
    res2 = cov_kernel(fd)

    np.testing.assert_allclose(
        res1,
        res2,
        atol=1e-7,
    )


def test_raises(
    fetch_weather_subset: FDataGrid,
    covariances_raise_fixture: Any,
) -> None:
    """Check raises ValueError.

    Check that non-functional kernels raise a ValueError exception
    with functional data.
    """
    fd = fetch_weather_subset
    cov_kernel = covariances_raise_fixture

    pytest.raises(
        ValueError,
        cov_kernel,
        fd,
    )


def test_precalc_example(
    precalc_example_data: list[  # noqa: WPS320
        FDataBasis, FDataBasis, cov.Covariance, np.array,
    ],
):
    """Check the precalculated example for Linear covariance kernel.

    Compare the theoretical precalculated results against the covariance kernel
    implementation, for different kernels.
    The structure of the input is a list containing:
        [First functional dataset, Second functional dataset,
        Covariance kernel used, Result]
    """
    fd1, fd2, kernel, precalc_result = precalc_example_data
    computed_result = kernel(fd1, fd2)
    np.testing.assert_allclose(
        computed_result,
        precalc_result,
        rtol=1e-6,
    )


def test_multivariate_covariance_kernel(
    multivariate_data: np.array,
    covariance_and_params: Any,
) -> None:
    """Test general covariance kernel against scikit-learn's kernel."""
    cov_kernel, param_dict = covariance_and_params
    for input_params in list(ParameterGrid(param_dict)):
        _test_compare_sklearn(multivariate_data, cov_kernel(**input_params))
