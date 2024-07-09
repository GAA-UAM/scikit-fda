"""Tests for FDM."""
from typing import Tuple

import numpy as np
import pytest
from sklearn import datasets

from skfda import FData, FDataBasis, FDataGrid
from skfda.misc.covariances import Gaussian
from skfda.misc.metrics import PairwiseMetric, l1_distance, l2_distance
from skfda.preprocessing.dim_reduction import FDM
from skfda.representation.basis import MonomialBasis
from skfda.typing._numpy import NDArrayFloat


def _discretize_fdatabasis(fd_basis: FDataBasis) -> FDataGrid:
    return fd_basis.to_grid(
        np.linspace(*fd_basis.basis.domain_range[0], 300),
    )


def rbf_kernel(fd: FData, sigma: float) -> NDArrayFloat:
    """Calculate the radial basis function kernel.

    Args:
        fd: Functional data object
        sigma: locality parameter

    Returns:
        Computed real value
    """
    norm_matrix = PairwiseMetric(l2_distance)(fd)
    return np.exp(  # type: ignore[no-any-return]
        - np.square(norm_matrix) / (2 * sigma ** 2),
    )


def laplacian_kernel(fd: FData, sigma: float) -> NDArrayFloat:
    """Calculate the radial basis function kernel.

    Args:
        fd: Functional data object
        sigma: locality parameter

    Returns:
        Computed real value
    """
    norm_matrix = PairwiseMetric(l1_distance)(fd)
    return np.exp(- norm_matrix / sigma)  # type: ignore[no-any-return]


##############################################################################
# Fixtures
##############################################################################
dummy_fdata = FDataGrid(data_matrix=[[0.9]])


@pytest.fixture(
    params=[
        FDM(kernel=Gaussian(), alpha=-1),
        FDM(kernel=Gaussian(), n_components=0),
        FDM(kernel=Gaussian(), n_components=2),
        FDM(kernel=Gaussian(), n_steps=0),
    ],
)
def data_grid_param_check(request: FDM) -> Tuple[FDM, FDataGrid]:
    """Fixture for testing parameter checks."""
    return request.param, dummy_fdata


@pytest.fixture(
    params=[
        FDataGrid(
            data_matrix=[
                [52, 93, 15],
                [72, 61, 21],
                [83, 87, 75],
                [75, 88, 24],
            ],
            grid_points=[0, 1 / 2, 1],
        ),
    ],
)
def precalculated_fdatagrid_example(
    request: FDataGrid,
) -> FDataGrid:
    """Fixture for loading a precalculated FDataGrid example."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        (
            FDataBasis(
                basis=MonomialBasis(domain_range=(0, 2), n_basis=3),
                coefficients=[
                    [0, 0, 1],
                    [1, 1, 0],
                    [0, 2, 0],
                ],
            ),
            FDataBasis(
                basis=MonomialBasis(domain_range=(0, 2), n_basis=3),
                coefficients=[
                    [-1, 1, 0],
                ],
            ),
        ),
    ],
)
def precalculated_fdatabasis_example(
    request: FDataBasis,
) -> Tuple[FDataBasis, FDataGrid, FDataBasis]:
    """Load FDataBasis example.

    Fixture for loading a prealculated FDataBasis exmaple and discretizing
    an FDataBasis into a FDataGrid object.
    """
    fd_basis: FDataBasis
    fd_out: FDataBasis
    fd_basis, fd_out = request.param
    return (
        fd_basis,
        _discretize_fdatabasis(fd_basis),
        fd_out,
    )


@pytest.fixture
def moons_functional_dataset() -> FDataGrid:
    """Fixture for loading moons dataset example."""
    n_grid_pts, n_samples = 100, 20
    data_moon, _ = datasets.make_moons(
        n_samples=n_samples,
        noise=0.0,
        random_state=0,
    )
    grid = np.linspace(-np.pi, np.pi, n_grid_pts)

    # Basis
    basis_1 = np.sin(4 * grid)
    basis_2 = (grid ** 2 + 2 * grid - 2)
    basis = np.array([basis_1, basis_2])

    # Generate functional data object
    data_matrix = np.array(data_moon) @ basis
    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid,
    )

##############################################################################
# Tests
##############################################################################


def test_param_check(data_grid_param_check: Tuple[FDM, FDataGrid]) -> None:
    """Check that invalid parameters in fit raise exception."""
    fdm, fd = data_grid_param_check

    pytest.raises(
        ValueError,
        fdm.fit,
        fd,
    )


def test_precalculated_grid_example(
    precalculated_fdatagrid_example: FDataGrid,
) -> None:
    """Compare the embedding in grid against the fda package."""
    fd = precalculated_fdatagrid_example
    fdm = FDM(
        n_components=2,
        kernel=Gaussian(length_scale=10),
        alpha=0.9,
        n_steps=2,
    )
    embedding = fdm.fit_transform(fd)

    expected_embedding = [
        [-0.0513176, 0.4572292],
        [0.6188756, -0.2537263],
        [-0.5478710, -0.3488652],
        [-0.0525144, 0.3176671],
    ]

    np.testing.assert_allclose(
        embedding,
        expected_embedding,
        atol=1e-7,
    )


def test_precalculated_basis_example(
    precalculated_fdatabasis_example: Tuple[FDataBasis, FDataGrid, FDataBasis],
) -> None:
    """Compare the embedding in basis and grid against the fda package."""
    fd_basis, fd_grid, _ = precalculated_fdatabasis_example

    fdm_basis = FDM(
        n_components=2,
        kernel=Gaussian(),
        alpha=1,
        n_steps=2,
    )
    embedding_basis = fdm_basis.fit_transform(fd_basis)

    fdm_grid = FDM(
        n_components=2,
        kernel=Gaussian(),
        alpha=1,
        n_steps=2,
    )
    embedding_grid = fdm_grid.fit_transform(fd_grid)

    expected_transition_matrix = [
        [0.52466, 0.20713, 0.26821],
        [0.21187, 0.47341, 0.31472],
        [0.27529, 0.31581, 0.40891],
    ]

    # Check transition matrix
    for tran_matrix in (
        fdm_basis.transition_matrix_,
        fdm_grid.transition_matrix_,
    ):
        np.testing.assert_allclose(
            tran_matrix,
            expected_transition_matrix,
            atol=1e-5,
        )

    # Check diffusion coordinates
    expected_embedding = [
        [0.0687, -0.00285],
        [-0.05426, -0.00648],
        [-0.01607, 0.00944],
    ]

    for embedding in (embedding_basis, embedding_grid):
        np.testing.assert_allclose(
            embedding,
            expected_embedding,
            atol=1e-5,
        )


def test_nystrom(
    precalculated_fdatabasis_example: Tuple[FDataBasis, FDataGrid, FDataBasis],
) -> None:
    """Test Nystrom  method.

    Compare the embedding of out-of-sample points in basis and grid
    against the fda package via the Nystrom method.
    """
    fd_basis, fd_grid, fd_out = precalculated_fdatabasis_example

    fdm_basis = FDM(
        n_components=2,
        kernel=Gaussian(),
        alpha=1,
        n_steps=2,
    )
    embedding_basis = fdm_basis.fit(fd_basis).transform(fd_out)

    fdm_grid = FDM(
        n_components=2,
        kernel=Gaussian(),
        alpha=1,
        n_steps=2,
    )
    embedding_grid = fdm_grid.fit(fd_grid).transform(
        _discretize_fdatabasis(fd_out),
    )

    expected_embedding = [
        [0.156125, -0.021132],
    ]

    np.testing.assert_allclose(
        embedding_basis,
        expected_embedding,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        embedding_grid,
        expected_embedding,
        atol=1e-3,
    )


def test_moons_dataset(
    moons_functional_dataset: FDataGrid,
) -> None:
    """Test the embedding for a small version of the moons dataset example."""
    fdata = moons_functional_dataset
    alpha, sigma = (1.0, 2.5)
    fdm = FDM(
        n_components=2,
        kernel=Gaussian(length_scale=sigma),
        alpha=alpha,
        n_steps=1,
    )
    embedding = fdm.fit_transform(fdata)

    expected_embedding = [
        [-0.07094005, -0.19475867],
        [0.04054447, -0.21670899],
        [0.09244529, -0.19569411],
        [0.07094005, -0.19475867],
        [0.13443593, -0.12554232],
        [-0.21055562, 0.01815767],
        [0.27732511, 0.1802182],
        [-0.26962763, 0.16043396],
        [0.29220798, 0.22086707],
        [0.18339034, -0.04185513],
        [0.29344042, 0.22420163],
        [-0.29220798, 0.22086707],
        [-0.09244529, -0.19569411],
        [0.21055562, 0.01815767],
        [-0.27732511, 0.1802182],
        [-0.04054447, -0.21670899],
        [0.26962763, 0.16043396],
        [-0.13443593, -0.12554232],
        [-0.29344042, 0.22420163],
        [-0.18339034, -0.04185513],
    ]

    np.testing.assert_allclose(
        embedding,
        expected_embedding,
        atol=1e-5,
    )


def test_gaussian_covariance(
    moons_functional_dataset: FDataGrid,
) -> None:
    """Test Gaussian covariance in covariance module against RBF kernel."""
    sigma = 0.1
    fdata = moons_functional_dataset
    gaussian_cov_matrix = Gaussian(length_scale=sigma)(fdata)
    rbf_kernel_matrix = rbf_kernel(fdata, sigma)

    np.testing.assert_allclose(
        gaussian_cov_matrix,
        rbf_kernel_matrix,
        atol=1e-7,
    )
