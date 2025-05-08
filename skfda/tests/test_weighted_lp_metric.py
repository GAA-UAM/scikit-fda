"""Tests for WeightedLpDistance and WeightedLpNorm."""

import math

import numpy as np
import pytest

from skfda import FDataBasis, FDataGrid
from skfda.datasets import make_multimodal_samples
from skfda.misc.metrics import weighted_lp_distance, weighted_lp_norm
from skfda.representation.basis import MonomialBasis


@pytest.fixture
def sample_data() -> dict:
    """Provide various test fixtures."""
    grid_points = [1, 2, 3, 4, 5]
    fd = FDataGrid(
        [
            [2, 3, 4, 5, 6],
            [1, 4, 9, 16, 25],
        ],
        grid_points=grid_points,
    )
    basis = MonomialBasis(n_basis=3, domain_range=(1, 5))
    fd_basis = FDataBasis(basis, [[1, 1, 0], [0, 0, 1]])
    fd_grid_in_basis = fd.to_basis(MonomialBasis(n_basis=4))
    fd_vector_valued = fd.concatenate(fd, as_coordinates=True)
    fd_surface = make_multimodal_samples(
        n_samples=3,
        dim_domain=2,
        random_state=0,
    )
    array = np.array(
        [
            [2, 3, 4, 5, 6],
            [1, 4, 9, 16, 25],
        ],
        dtype=float,
    )

    return {
        "fd": fd,
        "fd_basis": fd_basis,
        "fd_grid_in_basis": fd_grid_in_basis,
        "fd_vector_valued": fd_vector_valued,
        "fd_surface": fd_surface,
        "array": array,
    }


def test_lp_norm_grid(sample_data) -> None:
    """Test that the Lp norms work with FDataGrid."""
    fd = sample_data["fd"]

    np.testing.assert_allclose(weighted_lp_norm(fd, p=1), [16.0, 41.33333333])
    np.testing.assert_allclose(
        weighted_lp_norm(fd, p=2), [8.326664, 25.006666],
    )
    np.testing.assert_allclose(
        weighted_lp_norm(fd, p=3), [6.839904, 22.401268],
    )
    np.testing.assert_allclose(weighted_lp_norm(fd, p=math.inf), [6, 25])


def test_lp_norm_basis(sample_data) -> None:
    """Test that the L2 norm works with FDataBasis."""
    fd_basis = sample_data["fd_basis"]

    np.testing.assert_allclose(
        weighted_lp_norm(fd_basis, p=2), [8.326664, 24.996],
    )


def test_lp_norm_basis_equivalent(sample_data) -> None:
    """Test that the Lp norms in basis are similar to FDataGrid."""
    fd = sample_data["fd"]
    fd_grid_in_basis = sample_data["fd_grid_in_basis"]

    np.testing.assert_allclose(
        weighted_lp_norm(fd_grid_in_basis, p=1),
        weighted_lp_norm(fd, p=1),
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        weighted_lp_norm(fd_grid_in_basis, p=2),
        weighted_lp_norm(fd, p=2),
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        weighted_lp_norm(fd_grid_in_basis, p=3),
        weighted_lp_norm(fd, p=3),
        rtol=1e-2,
    )


def test_lp_norm_vector_valued(sample_data) -> None:
    """Test that the Lp norms work with vector-valued FDataGrid."""
    fd = sample_data["fd_vector_valued"]

    np.testing.assert_allclose(weighted_lp_norm(fd, p=1), [32.0, 82.666667])
    np.testing.assert_allclose(weighted_lp_norm(fd, p=math.inf), [6, 25])


def test_lp_norm_surface_inf(sample_data) -> None:
    """Test that the Linf norm works with multidimensional domains."""
    fd_surface = sample_data["fd_surface"]

    np.testing.assert_allclose(
        weighted_lp_norm(fd_surface, p=np.inf).round(5),
        [0.99994, 0.99793, 0.99868],
    )


def test_lp_norm_surface(sample_data) -> None:
    """Test the integration of surfaces."""
    fd_surface = sample_data["fd_surface"]

    np.testing.assert_allclose(
        weighted_lp_norm(fd_surface, p=1),
        [0.125663, 0.125637, 0.125661],
        rtol=1e-5,
    )


def test_lp_error_dimensions(sample_data) -> None:
    """Test error on metric between different kind of objects."""
    fd = sample_data["fd"]
    fd_surf = sample_data["fd_surface"]
    fd_vec = sample_data["fd_vector_valued"]

    with pytest.raises(ValueError):
        weighted_lp_distance(fd, fd_surf, p=2)

    with pytest.raises(ValueError):
        weighted_lp_distance(fd, fd_vec, p=2)

    with pytest.raises(ValueError):
        weighted_lp_distance(fd_surf, fd_vec, p=2)


def test_lp_error_domain_ranges(sample_data) -> None:
    """Test error on metric between objects with different domains."""
    fd = sample_data["fd"]
    fd2 = FDataGrid(
        [
            [2, 3, 4, 5, 6],
            [1, 4, 9, 16, 25],
        ],
        grid_points=[2, 3, 4, 5, 6],
    )

    with pytest.raises(ValueError):
        weighted_lp_distance(fd, fd2, p=2)


def test_lp_error_grid_points(sample_data) -> None:
    """Test error on metric for FDataGrids with different grid points."""
    fd = sample_data["fd"]
    fd2 = FDataGrid(
        [
            [2, 3, 4, 5, 6],
            [1, 4, 9, 16, 25],
        ],
        grid_points=[1, 2, 4, 4.3, 5],
    )

    with pytest.raises(ValueError):
        weighted_lp_distance(fd, fd2, p=2)


def test_lp_array(sample_data) -> None:
    """Test that the Lp norms work with arrays."""
    array = sample_data["array"]

    np.testing.assert_allclose(weighted_lp_norm(array, p=1), [20, 55])
    np.testing.assert_allclose(
        weighted_lp_norm(array, p=2), [9.48683298, 31.28897569],
    )
    np.testing.assert_allclose(
        weighted_lp_norm(array, p=3), [7.60590492, 27.37519199],
    )
    np.testing.assert_allclose(weighted_lp_norm(array, p=math.inf), [6, 25])
