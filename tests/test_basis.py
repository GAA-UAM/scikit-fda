import unittest

import numpy as np

from fda.basis import FDataBasis, Monomial, BSpline, Fourier


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_from_data_cholesky(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), nbasis=5)
        np.testing.assert_array_equal(
            FDataBasis.from_data(x, t, basis, smoothness_parameter=10,
                                 penalty_degree=2, method='cholesky'
                                 ).coefficients.round(2),
            np.array([[0.60, 0.47, 0.20, -0.07, -0.20]])
        )

    def test_from_data_qr(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), nbasis=5)
        np.testing.assert_array_equal(
            FDataBasis.from_data(x, t, basis, smoothness_parameter=10,
                                 penalty_degree=2, method='qr'
                                 ).coefficients.round(2),
            np.array([[0.60, 0.47, 0.20, -0.07, -0.20]])
        )

    def test_monomial_smoothing(self):
        # It does not have much sense to apply smoothing in this basic case
        # where the fit is very good but its just for testing purposes
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = Monomial(nbasis=4)
        fd = FDataBasis.from_data(x, t, basis,
                                  penalty_degree=2,
                                  smoothness_parameter=1)
        # These results where extracted from the R package fda
        np.testing.assert_array_equal(
            fd.coefficients.round(2), np.array([[0.61, -0.88, 0.06, 0.02]]))

    def test_bspline_penalty_special_case(self):
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

    def test_basis_basis_inprod(self):
        monomial = Monomial(nbasis=5)
        bspline = BSpline(nbasis=5, order=4)
        np.testing.assert_array_equal(
            bspline.inprod(monomial).round(3),
            np.array([[0.125, 0.25, 0.25, 0.25, 0.125],
                      [0.012, 0.075, 0.125, 0.175, 0.113],
                      [0.002, 0.029, 0.071, 0.129, 0.102],
                      [0., 0.013, 0.044, 0.099, 0.093],
                      [0., 0.007, 0.029, 0.078, 0.086]])
        )

    def test_basis_fdatabasis_inprod(self):
        monomial = Monomial(nbasis=3)
        bspline = BSpline(nbasis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_array_equal(
            bsplinefd.inprod(monomial).round(3),
            np.array([[2., 7., 12.],
                      [1.296, 3.796, 6.296],
                      [0.963, 2.63, 4.296]])
        )

    def test_fdatabasis_fdatabasis_inprod(self):
        monomial = Monomial(nbasis=4)
        monomialfd = FDataBasis(monomial, [[5, 4, 1, 0],
                                           [4, 2, 1, 0],
                                           [4, 1, 6, 4],
                                           [4, 5, 0, 1],
                                           [5, 6, 2, 0]])
        bspline = BSpline(nbasis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_array_equal(
            bsplinefd.inprod(monomialfd).round(3),
            np.array([[16.148, 52.815, 89.481, 11.556, 38.222],
                      [64.889, 18.147, 55.647, 93.147, 15.25],
                      [49., 82.75, 19.704, 63.037, 106.37]])
        )

    def test_comutativity_inprod(self):
        monomial = Monomial(nbasis=3)
        bspline = BSpline(nbasis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_array_equal(
            bsplinefd.inprod(monomial).round(3),
            np.transpose(monomial.inprod(bsplinefd).round(3))
        )


if __name__ == '__main__':
    print()
    unittest.main()
