"""Test stats functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from skfda import FDataGrid
from skfda.typing._numpy import NDArrayFloat
from skfda.datasets import make_gaussian_process
from skfda.exploratory.stats import std
from skfda.misc.covariances import Gaussian
from skfda.representation.basis import FourierBasis


@pytest.fixture(params=[61, 71])
def n_basis(request: Any) -> int:
    """Fixture for n_basis to test."""
    return request.param


@pytest.fixture
def start() -> int:
    """Fixture for the infimum of the domain."""
    return 0


@pytest.fixture
def stop() -> int:
    """Fixture for the supremum of the domain."""
    return 1


@pytest.fixture
def n_features() -> int:
    """Fixture for the number of features."""
    return 1000


@pytest.fixture
def gaussian_process(start: int, stop: int, n_features: int) -> FDataGrid:
    """Fixture for a Gaussian process."""
    return make_gaussian_process(
        start=start,
        stop=stop,
        n_samples=100,
        n_features=n_features,
        mean=0.0,
        cov=Gaussian(variance=1, length_scale=0.1),
        random_state=0,
    )


def test_std_gaussian_fourier(
    start: int,
    stop: int,
    n_features: int,
    n_basis: int,
    gaussian_process: FDataGrid,
) -> None:
    """Test standard deviation: Gaussian processes and a Fourier basis."""
    fourier_basis = FourierBasis(n_basis=n_basis, domain_range=(0, 1))
    fd = gaussian_process.to_basis(fourier_basis)

    std_fd = std(fd)
    grid = np.linspace(start, stop, n_features)
    almost_std_fd = std(fd.to_grid(grid)).to_basis(fourier_basis)

    inner_grid_limit = n_features // 10
    inner_grid = grid[inner_grid_limit:-inner_grid_limit]
    np.testing.assert_allclose(
        std_fd(inner_grid),
        almost_std_fd(inner_grid),
        rtol=1e-3,
    )

    outer_grid = grid[:inner_grid_limit] + grid[-inner_grid_limit:]
    np.testing.assert_allclose(
        std_fd(outer_grid),
        almost_std_fd(outer_grid),
        rtol=1e-2,
    )


@pytest.mark.parametrize("fdatagrid, expected_std_data_matrix", [
    (
        FDataGrid(
            data_matrix=[
                [[0, 1, 2, 3, 4, 5], [0, -1, -2, -3, -4, -5]],
                [[2, 3, 4, 5, 6, 7], [-2, -3, -4, -5, -6, -7]],
            ],
            grid_points=[
                [-2, -1],
                [0, 1, 2, 3, 4, 5]
            ],
        ),
        np.full((1, 2, 6, 1), np.sqrt(2))
    ),
])
def test_std_fdatagrid(
    fdatagrid: FDataGrid,
    expected_std_data_matrix: NDArrayFloat,
) -> None:
    """Test some FDataGrids' stds."""
    np.testing.assert_allclose(
        std(fdatagrid).data_matrix,
        expected_std_data_matrix,
    )
