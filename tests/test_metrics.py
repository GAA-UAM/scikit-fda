"""Tests for the metrics module."""

import unittest

import numpy as np

from skfda import FDataBasis, FDataGrid
from skfda.datasets import make_multimodal_samples
from skfda.misc.metrics import l2_distance, lp_norm
from skfda.representation.basis import Monomial


class TestLpMetrics(unittest.TestCase):
    """Test the lp metrics."""

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
        basis = Monomial(n_basis=3, domain_range=(1, 5))
        self.fd_basis = FDataBasis(basis, [[1, 1, 0], [0, 0, 1]])
        self.fd_curve = self.fd.concatenate(self.fd, as_coordinates=True)
        self.fd_surface = make_multimodal_samples(
            n_samples=3,
            dim_domain=2,
            random_state=0,
        )

    def test_lp_norm(self) -> None:

        np.testing.assert_allclose(lp_norm(self.fd, p=1), [16.0, 41.33333333])
        np.testing.assert_allclose(lp_norm(self.fd, p=np.inf), [6, 25])

    def test_lp_norm_curve(self) -> None:

        np.testing.assert_allclose(
            lp_norm(self.fd_curve, p=1),
            [32.0, 82.666667],
        )
        np.testing.assert_allclose(
            lp_norm(self.fd_curve, p=np.inf),
            [6, 25],
        )

    def test_lp_norm_surface_inf(self) -> None:
        np.testing.assert_allclose(
            lp_norm(self.fd_surface, p=np.inf).round(5),
            [0.99994, 0.99793, 0.99868],
        )

    def test_lp_norm_surface(self) -> None:
        # Integration of surfaces not implemented, add test case after
        # implementation
        self.assertEqual(lp_norm(self.fd_surface, p=1), NotImplemented)

    def test_lp_error_dimensions(self) -> None:
        # Case internal arrays
        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, self.fd_surface)

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd, self.fd_curve)

        with np.testing.assert_raises(ValueError):
            l2_distance(self.fd_surface, self.fd_curve)

    def test_lp_error_domain_ranges(self) -> None:
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
