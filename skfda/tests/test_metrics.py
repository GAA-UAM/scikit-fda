"""Tests for the metrics module."""

import unittest

import numpy as np

from skfda import FDataBasis, FDataGrid
from skfda.datasets import make_multimodal_samples
from skfda.misc.metrics import (
    l1_norm,
    l2_distance,
    l2_norm,
    linf_norm,
    lp_norm,
)
from skfda.representation.basis import MonomialBasis


class TestLp(unittest.TestCase):
    """Test the lp norms and distances."""

    def setUp(self) -> None:
        """Create a few functional data objects."""
        grid_points = [1, 2, 3, 4, 5]
        self.fd = FDataGrid(
            [
                [2, 3, 4, 5, 6],
                [1, 4, 9, 16, 25],
            ],
            grid_points=grid_points,
        )
        basis = MonomialBasis(n_basis=3, domain_range=(1, 5))
        self.fd_basis = FDataBasis(basis, [[1, 1, 0], [0, 0, 1]])
        self.fd_vector_valued = self.fd.concatenate(
            self.fd,
            as_coordinates=True,
        )
        self.fd_surface = make_multimodal_samples(
            n_samples=3,
            dim_domain=2,
            random_state=0,
        )

    def test_lp_norm_grid(self) -> None:
        """Test that the Lp norms work with FDataGrid."""
        np.testing.assert_allclose(
            l1_norm(self.fd),
            [16.0, 41.33333333],
        )

        np.testing.assert_allclose(
            l2_norm(self.fd),
            [8.326664, 25.006666],
        )

        np.testing.assert_allclose(
            lp_norm(self.fd, p=3),
            [6.839904, 22.401268],
        )

        np.testing.assert_allclose(
            linf_norm(self.fd),
            [6, 25],
        )

    def test_lp_norm_basis(self) -> None:
        """Test that the L2 norm works with FDataBasis."""
        np.testing.assert_allclose(
            l2_norm(self.fd_basis),
            [8.326664, 24.996],
        )

    def test_lp_norm_vector_valued(self) -> None:
        """Test that the Lp norms work with vector-valued FDataGrid."""
        np.testing.assert_allclose(
            l1_norm(self.fd_vector_valued),
            [32.0, 82.666667],
        )
        np.testing.assert_allclose(
            linf_norm(self.fd_vector_valued),
            [6, 25],
        )

    def test_lp_norm_surface_inf(self) -> None:
        """Test that the Linf norm works with multidimensional domains."""
        np.testing.assert_allclose(
            lp_norm(self.fd_surface, p=np.inf).round(5),
            [0.99994, 0.99793, 0.99868],
        )

    def test_lp_norm_surface(self) -> None:
        """Test that integration of surfaces has not been implemented."""
        # Integration of surfaces not implemented, add test case after
        # implementation
        self.assertEqual(lp_norm(self.fd_surface, p=1), NotImplemented)

    def test_lp_error_dimensions(self) -> None:
        """Test error on metric between different kind of objects."""
        # Case internal arrays
        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, self.fd_surface)

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, self.fd_vector_valued)

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd_surface, self.fd_vector_valued)

    def test_lp_error_domain_ranges(self) -> None:
        """Test error on metric between objects with different domains."""
        grid_points = [2, 3, 4, 5, 6]
        fd2 = FDataGrid(
            [
                [2, 3, 4, 5, 6],
                [1, 4, 9, 16, 25],
            ],
            grid_points=grid_points,
        )

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, fd2)

    def test_lp_error_grid_points(self) -> None:
        """Test error on metric for FDataGrids with different grid points."""
        grid_points = [1, 2, 4, 4.3, 5]
        fd2 = FDataGrid(
            [
                [2, 3, 4, 5, 6],
                [1, 4, 9, 16, 25],
            ],
            grid_points=grid_points,
        )

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, fd2)


if __name__ == '__main__':
    unittest.main()
