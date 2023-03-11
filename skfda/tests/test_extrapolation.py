"""Test to check the extrapolation module."""

import unittest

import numpy as np

from skfda import FDataBasis, FDataGrid
from skfda.datasets import make_sinusoidal_process
from skfda.representation.basis import FourierBasis
from skfda.representation.extrapolation import (
    BoundaryExtrapolation,
    ExceptionExtrapolation,
    FillExtrapolation,
    PeriodicExtrapolation,
)


class TestExtrapolation(unittest.TestCase):
    """Extrapolation tests."""

    def setUp(self) -> None:
        """Create example data."""
        self.grid = make_sinusoidal_process(n_samples=2, random_state=0)
        self.basis = self.grid.to_basis(FourierBasis())
        self.dummy_data = [[1, 2, 3], [2, 3, 4]]

    def test_constructor_fdatabasis_setting(self) -> None:
        """Check argument normalization in constructor for FDataBasis."""
        coeff = self.dummy_data
        basis = FourierBasis(n_basis=3)

        a = FDataBasis(basis, coeff)
        self.assertEqual(a.extrapolation, None)

        a = FDataBasis(basis, coeff, extrapolation="periodic")
        self.assertEqual(a.extrapolation, PeriodicExtrapolation())

        self.assertNotEqual(a.extrapolation, BoundaryExtrapolation())

        a = FDataBasis(basis, coeff, extrapolation=BoundaryExtrapolation())
        self.assertEqual(a.extrapolation, BoundaryExtrapolation())

        a = FDataBasis(basis, coeff, extrapolation="exception")
        self.assertEqual(a.extrapolation, ExceptionExtrapolation())

        a = FDataBasis(basis, coeff, extrapolation=FillExtrapolation(0))
        self.assertEqual(a.extrapolation, FillExtrapolation(0))
        self.assertNotEqual(a.extrapolation, FillExtrapolation(1))

    def test_constructor_fdatagrid_setting(self) -> None:
        """Check argument normalization in constructor for FDataGrid."""
        data = self.dummy_data

        a = FDataGrid(data)
        self.assertEqual(a.extrapolation, None)

        a = FDataGrid(data, extrapolation="periodic")
        self.assertEqual(a.extrapolation, PeriodicExtrapolation())

        a = FDataGrid(data, extrapolation=BoundaryExtrapolation())
        self.assertEqual(a.extrapolation, BoundaryExtrapolation())

        self.assertNotEqual(a.extrapolation, ExceptionExtrapolation())

        a = FDataGrid(data, extrapolation="exception")
        self.assertEqual(a.extrapolation, ExceptionExtrapolation())

        a = FDataGrid(data, extrapolation=FillExtrapolation(0))
        self.assertEqual(a.extrapolation, FillExtrapolation(0))
        self.assertNotEqual(a.extrapolation, FillExtrapolation(1))

    def test_setting(self) -> None:
        """Check argument in setter."""
        data = self.dummy_data
        a = FDataGrid(data)

        a.extrapolation = PeriodicExtrapolation()
        self.assertEqual(a.extrapolation, PeriodicExtrapolation())

        a.extrapolation = "bounds"  # type: ignore[assignment]
        self.assertEqual(a.extrapolation, BoundaryExtrapolation())

        a.extrapolation = ExceptionExtrapolation()
        self.assertEqual(a.extrapolation, ExceptionExtrapolation())

        a.extrapolation = "zeros"  # type: ignore[assignment]
        self.assertEqual(a.extrapolation, FillExtrapolation(0))

        self.assertNotEqual(a.extrapolation, FillExtrapolation(1))

    def test_periodic(self) -> None:
        """Test periodic extrapolation."""
        self.grid.extrapolation = PeriodicExtrapolation()
        data = self.grid([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [-0.723516, 0.976450, -0.723516],
                [-1.085999, 0.759385, -1.085999],
            ],
            rtol=1e-6,
        )

        self.basis.extrapolation = "periodic"  # type: ignore[assignment]
        data = self.basis([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [-0.690170, 0.691735, -0.690170],
                [-1.020821, 1.056383, -1.020821],
            ],
            rtol=1e-6,
        )

    def test_boundary(self) -> None:
        """Test boundary-copying extrapolation."""
        self.grid.extrapolation = "bounds"  # type: ignore[assignment]
        data = self.grid([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [0.976450, 0.976450, 0.796817],
                [0.759385, 0.759385, 1.125063],
            ],
            rtol=1e-6,
        )

        self.basis.extrapolation = "bounds"  # type: ignore[assignment]
        data = self.basis([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [0.691735, 0.691735, 0.691735],
                [1.056383, 1.056383, 1.056383],
            ],
            rtol=1e-6,
        )

    def test_exception(self) -> None:
        """Test no extrapolation (exception)."""
        self.grid.extrapolation = "exception"  # type: ignore[assignment]

        with np.testing.assert_raises(ValueError):
            self.grid([-0.5, 0, 1.5])

        self.basis.extrapolation = "exception"  # type: ignore[assignment]

        with np.testing.assert_raises(ValueError):
            self.basis([-0.5, 0, 1.5])

    def test_zeros(self) -> None:
        """Test zeros extrapolation."""
        self.grid.extrapolation = "zeros"  # type: ignore[assignment]
        data = self.grid([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [0, 0.976450, 0],
                [0, 0.759385, 0],
            ],
            rtol=1e-6,
        )

        self.basis.extrapolation = "zeros"  # type: ignore[assignment]
        data = self.basis([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [0, 0.691735, 0],
                [0, 1.056383, 0],
            ],
            rtol=1e-6,
        )

    def test_nan(self) -> None:
        """Test nan extrapolation."""
        self.grid.extrapolation = "nan"  # type: ignore[assignment]
        data = self.grid([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [np.nan, 0.976450, np.nan],
                [np.nan, 0.759385, np.nan],
            ],
            rtol=1e-6,
        )

        self.basis.extrapolation = "nan"  # type: ignore[assignment]
        data = self.basis([-0.5, 0, 1.5])

        np.testing.assert_allclose(
            data[..., 0],
            [
                [np.nan, 0.691735, np.nan],
                [np.nan, 1.056383, np.nan],
            ],
            rtol=1e-6,
        )


if __name__ == '__main__':
    unittest.main()
