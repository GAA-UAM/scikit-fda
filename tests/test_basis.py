import unittest
from fda.basis import Basis, FDataBasis, Constant, Monomial, BSpline, Fourier
import numpy as np


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

    def test_basis_product_generic(self):
        monomial = Monomial(nbasis=5)
        fourier = Fourier(nbasis=3)
        prod = BSpline(nbasis=9, order=8)
        self.assertEqual(Basis.default_basis_of_product(monomial, fourier), prod)

    def test_basis_constant_product(self):
        constant = Constant()
        monomial = Monomial()
        fourier = Fourier()
        bspline = BSpline(nbasis=5, order=3)
        self.assertEqual(constant.basis_of_product(monomial), monomial)
        self.assertEqual(constant.basis_of_product(fourier), fourier)
        self.assertEqual(constant.basis_of_product(bspline), bspline)
        self.assertEqual(monomial.basis_of_product(constant), monomial)
        self.assertEqual(fourier.basis_of_product(constant), fourier)
        self.assertEqual(bspline.basis_of_product(constant), bspline)

    def test_basis_fourier_product(self):
        # Test when periods are the same
        fourier = Fourier(nbasis=5)
        fourier2 = Fourier(nbasis=3)
        prod = Fourier(nbasis=7)
        self.assertEqual(fourier.basis_of_product(fourier2), prod)

        # Test when periods are different
        fourier2 = Fourier(nbasis=3, period=2)
        prod = BSpline(nbasis=9, order=8)
        self.assertEqual(fourier.basis_of_product(fourier2), prod)

    def test_basis_monomial_product(self):
        monomial = Monomial(nbasis=5)
        monomial2 = Monomial(nbasis=3)
        prod = Monomial(nbasis=8)
        self.assertEqual(monomial.basis_of_product(monomial2), prod)

    def test_basis_bspline_product(self):
        bspline = BSpline(nbasis=6, order=4)
        bspline2 = BSpline(domain_range=(0, 1), nbasis=6, order=4, knots=[0, 0.3, 1 / 3, 1])
        prod = BSpline(domain_range=(0,1), nbasis=10, order=7, knots=[0, 0.3, 1/3, 2/3,1])
        self.assertEqual(bspline.basis_of_product(bspline2), prod)


if __name__ == '__main__':
    print()
    unittest.main()
