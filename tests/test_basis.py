import unittest
from fda.basis import FDataBasis, Monomial, BSpline, Fourier
import numpy as np


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_monomial_smoothing(self):
        # It does not have much sense to apply smoothing in this basic case
        # where the fit is very good but its just for testing purposes
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = Monomial(nbasis=4)
        fd = FDataBasis.from_data(x, t, basis,
                                  differential_operator=2,
                                  smoothness_parameter=1)
        # These results where extracted from the R package fda
        np.testing.assert_array_equal(
            fd.coefficients.round(2), np.array([[0.61, -0.88, 0.06, 0.02]]))

    def test_bspline_penalty(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_equal(
            basis.penalty(basis.order - 1),
            np.array([[1152., -2016., 1152., -288., 0.],
                      [-2016., 3600., -2304., 1008., -288.],
                      [1152., -2304., 2304., -2304., 1152.],
                      [-288., 1008., -2304., 3600., -2016.],
                      [0., -288., 1152., -2016., 1152.]]))

    def test_fourier_penalty(self):
        basis = Fourier(nbasis=5)
        np.testing.assert_array_equal(
            basis.penalty(2).round(2),
            np.array([[0., 0., 0., 0., 0.],
                      [0., 1558.55, 0., 0., 0.],
                      [0., 0., 1558.55, 0., 0.],
                      [0., 0., 0., 24936.73, 0.],
                      [0., 0., 0., 0., 24936.73]]))

    def test_bspline_penalty(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_equal(
            basis.penalty(2).round(2),
            np.array([[96., -132., 24., 12., 0.],
                      [-132., 192., -48., -24., 12.],
                      [24., -48., 48., -48., 24.],
                      [12., -24., -48., 192., -132.],
                      [0., 12., 24., -132., 96.]]))

    def test_bspline_penalty_numerical(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_equal(
            basis.penalty(coefficients=[0, 0, 1]).round(2),
            np.array([[96., -132., 24., 12., 0.],
                      [-132., 192., -48., -24., 12.],
                      [24., -48., 48., -48., 24.],
                      [12., -24., -48., 192., -132.],
                      [0., 12., 24., -132., 96.]]))


if __name__ == '__main__':
    print()
    unittest.main()
