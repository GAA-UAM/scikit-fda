"""Tests for Covariance module."""
from typing import Any, Tuple

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid

import skfda.misc.covariances as cov
from skfda import FDataBasis, FDataGrid
from skfda.datasets import fetch_phoneme
from skfda.representation.basis import MonomialBasis


def _test_compare_sklearn(
    multivariate_data: Any,
    cov: cov.Covariance,
) -> None:
    cov_sklearn = cov.to_sklearn()
    cov_matrix = cov(multivariate_data)
    cov_sklearn_matrix = cov_sklearn(multivariate_data)

    np.testing.assert_array_almost_equal(cov_matrix, cov_sklearn_matrix)

##############################################################################
# Fixtures
##############################################################################


@pytest.fixture
def fetch_phoneme_fixture() -> FDataGrid:
    """Fixture for loading the phoneme dataset example."""
    fd, _ = fetch_phoneme(return_X_y=True)
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


@pytest.fixture
def fdatabasis_data() -> Tuple[FDataBasis, FDataBasis]:
    """Fixture for getting fdatabasis objects."""
    basis = MonomialBasis(
        n_basis=2,
        domain_range=(-2, 2),
    )

    fd1 = FDataBasis(
        basis=basis,
        coefficients=[
            [1, 0],
            [1, 2],
        ],
    )

    fd2 = FDataBasis(
        basis=basis,
        coefficients=[
            [0, 1],
        ],
    )

    return fd1, fd2


@pytest.fixture
def multivariate_data() -> np.array:
    """Fixture for getting multivariate data."""
    return np.linspace(-1, 1, 1000)[:, np.newaxis]


@pytest.fixture(
    params=[
        (cov.Linear,
         {
             "variance": [1, 2],
             "intercept": [3, 4],
         },
         ),
        (cov.Polynomial,
         {
             "variance": [2],
             "intercept": [0, 2],
             "slope": [1, 2],
             "degree": [1, 2, 3],
         },
         ),
        (cov.Exponential,
         {
             "variance": [1, 2],
             "length_scale": [0.5, 1, 2],
         },
         ),
        (cov.Gaussian,
         {
             "variance": [1, 2],
             "length_scale": [0.5, 1, 2],
         },
         ),
        (cov.Matern,
         {
             "variance": [2],
             "length_scale": [0.5],
             "nu": [0.5, 1, 1.5, 2.5, 3.5, np.inf],
         },
         ),
    ],
)
def covariance_and_params(request: Any) -> Any:
    """Fixture to load the covariance functions."""
    return request.param


##############################################################################
# Tests
##############################################################################


def test_covariances(
    fetch_phoneme_fixture: Any,
    covariances_fixture: Any,
) -> None:
    """Check that invalid parameters in fit raise exception."""
    fd = fetch_phoneme_fixture
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
    fetch_phoneme_fixture: Any,
    covariances_raise_fixture: Any,
) -> None:
    """Check that it raises a ValueError exception."""
    fd = fetch_phoneme_fixture
    cov_kernel = covariances_raise_fixture

    np.testing.assert_raises(
        ValueError,
        cov_kernel,
        fd,
    )


def test_fdatabasis_example_linear(
    fdatabasis_data: Any,
) -> None:
    """Check a precalculated example for Linear covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Linear(variance=1 / 2, intercept=3)(fd1, fd2)
    res2 = np.array([[3 / 2], [3 / 2 + 32 / 6]])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_fdatabasis_example_polynomial(
    fdatabasis_data: Any,
) -> None:
    """Check a precalculated example for Polynomial covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Polynomial(
        variance=1 / 3,
        slope=2,
        intercept=1,
        degree=2,
    )(fd1, fd2)
    res2 = np.array([[1 / 3], [67**2 / 3**3]])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_fdatabasis_example_gaussian(
    fdatabasis_data: Any,
) -> None:
    """Check a precalculated example for Gaussian covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Gaussian(variance=3, length_scale=2)(fd1, fd2)
    res2 = np.array([
        [3 * np.exp(-7 / 6)],
        [3 * np.exp(-7 / 6)],
    ])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_fdatabasis_example_exponential(
    fdatabasis_data: Any,
) -> None:
    """Check a precalculated example for Exponential covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Exponential(variance=4, length_scale=5)(fd1, fd2)
    res2 = np.array([
        [4 * np.exp(-np.sqrt(28 / 3) / 5)],
        [4 * np.exp(-np.sqrt(28 / 3) / 5)],
    ])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_fdatabasis_example_matern(
    fdatabasis_data: Any,
) -> None:
    """Check a precalculated example for Matern covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Matern(variance=2, length_scale=3, nu=2)(fd1, fd2)
    res2 = np.array([
        [(2 / 3) ** 2 * (28 / 3) * 0.239775899566],
        [(2 / 3) ** 2 * (28 / 3) * 0.239775899566],
    ])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_multivariate_covariance_kernel(
    multivariate_data: Any,
    covariance_and_params: Any,
) -> None:
    """Test general covariance kernel against scikit-learn's kernel."""
    cov_kernel, param_dict = covariance_and_params
    for input_params in list(ParameterGrid(param_dict)):
        _test_compare_sklearn(multivariate_data, cov_kernel(**input_params))
