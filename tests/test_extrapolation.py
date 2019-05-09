"""Test to check the extrapolation module"""

import unittest

import numpy as np

from skfda.datasets import make_sinusoidal_process
from skfda.representation.basis import Fourier
from skfda import FDataGrid, FDataBasis
from skfda.representation.extrapolation import (
    PeriodicExtrapolation, BoundaryExtrapolation, ExceptionExtrapolation,
    FillExtrapolation)


class TestBasis(unittest.TestCase):

    def setUp(self):
        self.grid = make_sinusoidal_process(n_samples=2, random_state=0)
        self.basis = self.grid.to_basis(Fourier())
        self.dummy_data = [[1, 2, 3], [2, 3, 4]]

    def test_constructor_FDataBasis_setting(self):
        coeff = self.dummy_data
        basis = Fourier(nbasis=3)

        a = FDataBasis(basis, coeff)
        np.testing.assert_equal(a.extrapolation, None)

        a = FDataBasis(basis, coeff, extrapolation=PeriodicExtrapolation())
        np.testing.assert_equal(a.extrapolation, PeriodicExtrapolation())

        np.testing.assert_equal(
            a.extrapolation == BoundaryExtrapolation(), False)

        a = FDataBasis(basis, coeff, extrapolation=BoundaryExtrapolation())
        np.testing.assert_equal(a.extrapolation, BoundaryExtrapolation())

        a = FDataBasis(basis, coeff, extrapolation=ExceptionExtrapolation())
        np.testing.assert_equal(a.extrapolation, ExceptionExtrapolation())

        a = FDataBasis(basis, coeff, extrapolation=FillExtrapolation(0))
        np.testing.assert_equal(a.extrapolation, FillExtrapolation(0))

        np.testing.assert_equal(a.extrapolation == FillExtrapolation(1), False)

    def test_FDataBasis_setting(self):
        coeff = self.dummy_data
        basis = Fourier(nbasis=3)
        a = FDataBasis(basis, coeff)

        a.extrapolation = "periodic"
        np.testing.assert_equal(a.extrapolation, PeriodicExtrapolation())

        a.extrapolation = "bounds"
        np.testing.assert_equal(a.extrapolation, BoundaryExtrapolation())

        a.extrapolation = "exception"
        np.testing.assert_equal(a.extrapolation, ExceptionExtrapolation())

        a.extrapolation = "zeros"
        np.testing.assert_equal(a.extrapolation, FillExtrapolation(0))

        a.extrapolation = "nan"
        np.testing.assert_equal(a.extrapolation, FillExtrapolation(np.nan))

    def test_constructor_FDataGrid_setting(self):
        data = self.dummy_data

        a = FDataGrid(data)
        np.testing.assert_equal(a.extrapolation, None)

        a = FDataGrid(data, extrapolation=PeriodicExtrapolation())
        np.testing.assert_equal(a.extrapolation, PeriodicExtrapolation())

        a = FDataGrid(data, extrapolation=BoundaryExtrapolation())
        np.testing.assert_equal(a.extrapolation, BoundaryExtrapolation())

        np.testing.assert_equal(
            a.extrapolation == ExceptionExtrapolation(), False)

        a = FDataGrid(data, extrapolation=ExceptionExtrapolation())
        np.testing.assert_equal(a.extrapolation, ExceptionExtrapolation())

        a = FDataGrid(data, extrapolation=FillExtrapolation(0))
        np.testing.assert_equal(a.extrapolation, FillExtrapolation(0))
        np.testing.assert_equal(a.extrapolation == FillExtrapolation(1), False)

    def test_FDataGrid_setting(self):
        data = self.dummy_data
        a = FDataGrid(data)

        a.extrapolation = PeriodicExtrapolation()
        np.testing.assert_equal(a.extrapolation, PeriodicExtrapolation())

        a.extrapolation = "bounds"
        np.testing.assert_equal(a.extrapolation, BoundaryExtrapolation())

        a.extrapolation = "exception"
        np.testing.assert_equal(a.extrapolation, ExceptionExtrapolation())

        a.extrapolation = "zeros"
        np.testing.assert_equal(a.extrapolation, FillExtrapolation(0))

        np.testing.assert_equal(a.extrapolation == FillExtrapolation(1), False)

    def test_periodic(self):
        self.grid.extrapolation = PeriodicExtrapolation()
        data = self.grid([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[-0.724,  0.976, -0.724],
                                              [-1.086,  0.759, -1.086]])

        self.basis.extrapolation = "periodic"
        data = self.basis([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[-0.69,  0.692, -0.69],
                                              [-1.021,  1.056, -1.021]])

    def test_boundary(self):
        self.grid.extrapolation = "bounds"
        data = self.grid([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[0.976,  0.976,  0.797],
                                              [0.759,  0.759,  1.125]])

        self.basis.extrapolation = "bounds"
        data = self.basis([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[0.692, 0.692, 0.692],
                                              [1.056, 1.056, 1.056]])

    def test_exception(self):
        self.grid.extrapolation = "exception"

        with np.testing.assert_raises(ValueError):
            self.grid([-.5, 0, 1.5])

        self.basis.extrapolation = "exception"

        with np.testing.assert_raises(ValueError):
            self.basis([-.5, 0, 1.5])

    def test_zeros(self):
        self.grid.extrapolation = "zeros"
        data = self.grid([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[0.,  0.976,  0.],
                                              [0.,  0.759,  0.]])

        self.basis.extrapolation = "zeros"
        data = self.basis([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[0, 0.692, 0],
                                              [0, 1.056, 0]])

    def test_nan(self):
        self.grid.extrapolation = "nan"
        data = self.grid([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[np.nan,  0.976,  np.nan],
                                              [np.nan,  0.759,  np.nan]])

        self.basis.extrapolation = "nan"
        data = self.basis([-.5, 0, 1.5]).round(3)

        np.testing.assert_almost_equal(data, [[np.nan, 0.692, np.nan],
                                              [np.nan, 1.056, np.nan]])


if __name__ == '__main__':
    print()
    unittest.main()
