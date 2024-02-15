"""
Tests for Covariance module.

This file includes tests for multivariate data from the previous version
of the file. It additionally incorporates tests cases for functional data
objects.
"""
import pytest
from typing import Tuple

import numpy as np

import skfda.misc.covariances as cov
from skfda import FDataGrid, FDataBasis
from skfda.datasets import fetch_phoneme
from skfda.representation.basis import MonomialBasis


def _test_compare_sklearn(
    multivariate_data,
    cov: cov.Covariance,
) -> None:
    cov_sklearn = cov.to_sklearn()
    cov_matrix = cov(multivariate_data, multivariate_data)
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
def covariances_fixture(request) -> cov.Covariance:
    """Fixture for getting a covariance kernel function."""
    return request.param


@pytest.fixture(
    params=[
        cov.Brownian(),
        cov.WhiteNoise(),
    ],
)
def covariances_raise_fixture(request) -> cov.Covariance:
    """Fixture for getting a covariance kernel that raises a ValueError."""
    return request.param


@pytest.fixture
def fdatabasis_data() -> Tuple[FDataBasis, FDataBasis]:
    """Fixture for loading fdatabasis objects."""
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
def multivariate_data() -> None:
    """Fixture for loading multivariate data."""
    return np.linspace(-1, 1, 1000)[:, np.newaxis]


@pytest.fixture(
    params=[1, 2],
)
def variance_param(request) -> np.ndarray:
    """Fixture for loading variance parameter."""
    return request.param


@pytest.fixture(
    params=[0, 1, 2],
)
def intercept_param(request) -> np.ndarray:
    """Fixture for loading intercept parameter."""
    return request.param


@pytest.fixture(
    params=[1, 2],
)
def slope_param(request) -> np.ndarray:
    """Fixture for loading slope parameter."""
    return request.param


@pytest.fixture(
    params=[1, 2, 3],
)
def degree_param(request) -> np.ndarray:
    """Fixture for loading degree parameter."""
    return request.param


@pytest.fixture(
    params=[1, 2, 3],
)
def length_scale_param(request) -> np.ndarray:
    """Fixture for loading length scale parameter."""
    return request.param


@pytest.fixture(
    params=[0.5, 1, 1.5, 2.5, 3.5, np.inf],
)
def nu_param(request) -> np.ndarray:
    """Fixture for loading nu parameter."""
    return request.param


##############################################################################
# Tests
##############################################################################


def test_covariances(fetch_phoneme_fixture, covariances_fixture) -> None:
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


def test_raises(fetch_phoneme_fixture, covariances_raise_fixture):
    """Check that it raises a ValueError exception."""
    fd = fetch_phoneme_fixture
    cov_kernel = covariances_raise_fixture

    np.testing.assert_raises(
        ValueError,
        cov_kernel,
        fd,
    )


def test_fdatabasis_example_linear(fdatabasis_data):
    """Check a precalculated example for Linear covariance kernel."""
    fd1, fd2 = fdatabasis_data
    res1 = cov.Linear(variance=1 / 2, intercept=3)(fd1, fd2)
    res2 = np.array([[3 / 2], [3 / 2 + 32 / 6]])
    np.testing.assert_allclose(
        res1,
        res2,
        rtol=1e-6,
    )


def test_fdatabasis_example_polynomial(fdatabasis_data):
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


def test_fdatabasis_example_gaussian(fdatabasis_data):
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


def test_fdatabasis_example_exponential(fdatabasis_data):
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


def test_fdatabasis_example_matern(fdatabasis_data):
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


def test_multivariate_linear(
    multivariate_data,
    variance_param,
    intercept_param,
) -> None:
    """Test linear covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.Linear(
            variance=variance_param,
            intercept=intercept_param,
        ),
    )


def test_multivariate_polynomial(
    multivariate_data,
    variance_param,
    intercept_param,
    slope_param,
    degree_param,
) -> None:
    """Test polynomial covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.Polynomial(
            variance=variance_param,
            intercept=intercept_param,
            slope=slope_param,
            degree=degree_param,
        ),
    )


def test_multivariate_gaussian(
    multivariate_data,
    variance_param,
    length_scale_param,
) -> None:
    """Test gaussian covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.Gaussian(
            variance=variance_param,
            length_scale=length_scale_param,
        ),
    )


def test_multivariate_exponential(
    multivariate_data,
    variance_param,
    length_scale_param,
) -> None:
    """Test exponential covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.Exponential(
            variance=variance_param,
            length_scale=length_scale_param,
        ),
    )


def test_multivariate_matern(
    multivariate_data,
    variance_param,
    length_scale_param,
    nu_param,
) -> None:
    """Test matern covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.Matern(
            variance=variance_param,
            length_scale=length_scale_param,
            nu=nu_param,
        ),
    )


def test_multivariate_white_noise(
    multivariate_data,
    variance_param,
) -> None:
    """Test white noise covariance kernel against scikit learn's kernel."""
    _test_compare_sklearn(
        multivariate_data,
        cov.WhiteNoise(
            variance=variance_param,
        ),
    )
