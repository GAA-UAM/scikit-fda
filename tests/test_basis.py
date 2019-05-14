import unittest

from skfda.representation.basis import (Basis, FDataBasis, Constant, Monomial,
                                        BSpline, Fourier)

import numpy as np


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_from_data_cholesky(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), nbasis=5)
        np.testing.assert_array_almost_equal(
            FDataBasis.from_data(x, t, basis, smoothness_parameter=10,
                                 penalty_degree=2, method='cholesky'
                                 ).coefficients.round(2),
            np.array([[0.60, 0.47, 0.20, -0.07, -0.20]])
        )

    def test_from_data_qr(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), nbasis=5)
        np.testing.assert_array_almost_equal(
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
        np.testing.assert_array_almost_equal(
            fd.coefficients.round(2), np.array([[0.61, -0.88, 0.06, 0.02]]))

    def test_bspline_penalty_special_case(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_almost_equal(
            basis.penalty(basis.order - 1),
            np.array([[1152., -2016., 1152., -288., 0.],
                      [-2016., 3600., -2304., 1008., -288.],
                      [1152., -2304., 2304., -2304., 1152.],
                      [-288., 1008., -2304., 3600., -2016.],
                      [0., -288., 1152., -2016., 1152.]]))

    def test_fourier_penalty(self):
        basis = Fourier(nbasis=5)
        np.testing.assert_array_almost_equal(
            basis.penalty(2).round(2),
            np.array([[0., 0., 0., 0., 0.],
                      [0., 1558.55, 0., 0., 0.],
                      [0., 0., 1558.55, 0., 0.],
                      [0., 0., 0., 24936.73, 0.],
                      [0., 0., 0., 0., 24936.73]]))

    def test_bspline_penalty(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_almost_equal(
            basis.penalty(2).round(2),
            np.array([[96., -132., 24., 12., 0.],
                      [-132., 192., -48., -24., 12.],
                      [24., -48., 48., -48., 24.],
                      [12., -24., -48., 192., -132.],
                      [0., 12., 24., -132., 96.]]))

    def test_bspline_penalty_numerical(self):
        basis = BSpline(nbasis=5)
        np.testing.assert_array_almost_equal(
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

    def test_basis_basis_inprod(self):
        monomial = Monomial(nbasis=4)
        bspline = BSpline(nbasis=5, order=4)
        np.testing.assert_array_almost_equal(
            monomial.inner_product(bspline).round(3),
            np.array(
                [[0.12499983, 0.25000035, 0.24999965, 0.25000035, 0.12499983],
                 [0.01249991, 0.07500017, 0.12499983, 0.17500017, 0.11249991],
                 [0.00208338, 0.02916658, 0.07083342, 0.12916658, 0.10208338],
                 [0.00044654, 0.01339264, 0.04375022, 0.09910693, 0.09330368]])
            .round(3)
        )

    def test_basis_fdatabasis_inprod(self):
        monomial = Monomial(nbasis=4)
        bspline = BSpline(nbasis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_array_almost_equal(
            monomial.inner_product(bsplinefd).round(3),
            np.array([[2., 7., 12.],
                      [1.29626206, 3.79626206, 6.29626206],
                      [0.96292873, 2.62959539, 4.29626206],
                      [0.7682873, 2.0182873, 3.2682873]]).round(3)
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

        np.testing.assert_array_almost_equal(
            monomialfd.inner_product(bsplinefd).round(3),
            np.array([[16.14797697, 52.81464364, 89.4813103],
                      [11.55565285, 38.22211951, 64.88878618],
                      [18.14698361, 55.64698361, 93.14698361],
                      [15.2495976, 48.9995976, 82.7495976],
                      [19.70392982, 63.03676315, 106.37009648]]).round(3)
        )

    def test_comutativity_inprod(self):
        monomial = Monomial(nbasis=4)
        bspline = BSpline(nbasis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_array_almost_equal(
            bsplinefd.inner_product(monomial).round(3),
            np.transpose(monomial.inner_product(bsplinefd).round(3))
        )

    def test_fdatabasis_times_fdatabasis_fdatabasis(self):
        monomial = FDataBasis(Monomial(nbasis=3), [1, 2, 3])
        bspline = FDataBasis(BSpline(nbasis=6, order=4), [1, 2, 4, 1, 0, 1])
        times_fdar = monomial.times(bspline)

        prod_basis = BSpline(nbasis=9, order=6, knots=[0, 0.25, 0.5, 0.75, 1])
        prod_coefs = np.array([[0.9788352,  1.6289955,  2.7004969,  6.2678739,
                      8.7636441,  4.0069960,  0.7126961,  2.8826708,
                      6.0052311]])

        self.assertEqual(prod_basis, times_fdar.basis)
        np.testing.assert_array_almost_equal(prod_coefs, times_fdar.coefficients)

    def test_fdatabasis_times_fdatabasis_list(self):
        monomial = FDataBasis(Monomial(nbasis=3),
                              [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = monomial.times([3, 2, 1])

        expec_basis = Monomial(nbasis=3)
        expec_coefs = np.array([[3, 6, 9], [8, 10, 12], [7, 8, 9]])

        self.assertEqual(expec_basis, result.basis)
        np.testing.assert_array_almost_equal(expec_coefs, result.coefficients)

    def test_fdatabasis_times_fdatabasis_int(self):
        monomial = FDataBasis(Monomial(nbasis=3),
                              [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = monomial.times(3)

        expec_basis = Monomial(nbasis=3)
        expec_coefs = np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]])

        self.assertEqual(expec_basis, result.basis)
        np.testing.assert_array_almost_equal(expec_coefs, result.coefficients)

    def test_fdatabasis_derivative_constant(self):
        monomial = FDataBasis(Monomial(nbasis=8),
                              [1, 5, 8, 9, 7, 8, 4, 5])
        monomial2 = FDataBasis(Monomial(nbasis=5),
                              [[4, 9, 7, 4, 3],
                               [1, 7, 9, 8, 5],
                               [4, 6, 6, 6, 8]])

        np.testing.assert_equal(monomial.derivative(),
                                FDataBasis(Monomial(nbasis=7),
                                           [5, 16, 27, 28, 40, 24, 35]))
        np.testing.assert_equal(monomial.derivative(order=0), monomial)
        np.testing.assert_equal(monomial.derivative(order=6),
                                FDataBasis(Monomial(nbasis=2),
                                           [2880, 25200]))

        np.testing.assert_equal(monomial2.derivative(),
                                FDataBasis(Monomial(nbasis=4),
                                           [[9, 14, 12, 12],
                                            [7, 18, 24, 20],
                                            [6, 12, 18, 32]]))
        np.testing.assert_equal(monomial2.derivative(order=0), monomial2)
        np.testing.assert_equal(monomial2.derivative(order=3),
                                FDataBasis(Monomial(nbasis=2),
                                           [[24, 72],
                                            [48, 120],
                                            [36, 192]]))

    def test_fdatabasis_derivative_monomial(self):
        monomial = FDataBasis(Monomial(nbasis=8),
                              [1, 5, 8, 9, 7, 8, 4, 5])
        monomial2 = FDataBasis(Monomial(nbasis=5),
                              [[4, 9, 7, 4, 3],
                               [1, 7, 9, 8, 5],
                               [4, 6, 6, 6, 8]])

        np.testing.assert_equal(monomial.derivative(),
                                FDataBasis(Monomial(nbasis=7),
                                           [5, 16, 27, 28, 40, 24, 35]))
        np.testing.assert_equal(monomial.derivative(order=0), monomial)
        np.testing.assert_equal(monomial.derivative(order=6),
                                FDataBasis(Monomial(nbasis=2),
                                           [2880, 25200]))

        np.testing.assert_equal(monomial2.derivative(),
                                FDataBasis(Monomial(nbasis=4),
                                           [[9, 14, 12, 12],
                                            [7, 18, 24, 20],
                                            [6, 12, 18, 32]]))
        np.testing.assert_equal(monomial2.derivative(order=0), monomial2)
        np.testing.assert_equal(monomial2.derivative(order=3),
                                FDataBasis(Monomial(nbasis=2),
                                           [[24, 72],
                                            [48, 120],
                                            [36, 192]]))

    def test_fdatabasis_derivative_fourier(self):
        fourier = FDataBasis(Fourier(nbasis=7),
                              [1, 5, 8, 9, 8, 4, 5])
        fourier2 = FDataBasis(Fourier(nbasis=5),
                              [[4, 9, 7, 4, 3],
                               [1, 7, 9, 8, 5],
                               [4, 6, 6, 6, 8]])

        fou0 = fourier.derivative(order=0)
        fou1 = fourier.derivative()
        fou2 = fourier.derivative(order=2)

        np.testing.assert_equal(fou1.basis, fourier.basis)
        np.testing.assert_almost_equal(fou1.coefficients.round(5),
                                       np.atleast_2d([0, -50.26548, 31.41593,
                                                      -100.53096, 113.09734,
                                                      -94.24778, 75.39822]))
        np.testing.assert_equal(fou0, fourier)
        np.testing.assert_equal(fou2.basis, fourier.basis)
        np.testing.assert_almost_equal(fou2.coefficients.round(5),
                                       np.atleast_2d([0, -197.39209, -315.82734,
                                                      -1421.22303, -1263.30936,
                                                      -1421.22303, -1776.52879]))

        fou0 = fourier2.derivative(order=0)
        fou1 = fourier2.derivative()
        fou2 = fourier2.derivative(order=2)

        np.testing.assert_equal(fou1.basis, fourier2.basis)
        np.testing.assert_almost_equal(fou1.coefficients.round(5),
                                       [[0, -43.98230, 56.54867, -37.69911, 50.26548],
                                        [0, -56.54867, 43.98230, -62.83185, 100.53096],
                                        [0, -37.69911, 37.69911, -100.53096, 75.39822]])
        np.testing.assert_equal(fou0, fourier2)
        np.testing.assert_equal(fou2.basis, fourier2.basis)
        np.testing.assert_almost_equal(fou2.coefficients.round(5),
                                       [[0, -355.30576, -276.34892, -631.65468, -473.74101],
                                        [0, -276.34892, -355.30576, -1263.30936, -789.56835],
                                        [0, -236.87051, -236.87051, -947.48202, -1263.30936]])

    def test_fdatabasis_derivative_bspline(self):
        bspline = FDataBasis(BSpline(nbasis=8),
                             [1, 5, 8, 9, 7, 8, 4, 5])
        bspline2 = FDataBasis(BSpline(nbasis=5),
                              [[4, 9, 7, 4, 3],
                               [1, 7, 9, 8, 5],
                               [4, 6, 6, 6, 8]])

        bs0 = bspline.derivative(order=0)
        bs1 = bspline.derivative()
        bs2 = bspline.derivative(order=2)
        np.testing.assert_equal(bs1.basis, BSpline(nbasis=7, order=3))
        np.testing.assert_almost_equal(bs1.coefficients,
                                       np.atleast_2d([60, 22.5, 5,
                                                      -10, 5, -30, 15]))
        np.testing.assert_equal(bs0, bspline)
        np.testing.assert_equal(bs2.basis, BSpline(nbasis=6, order=2))
        np.testing.assert_almost_equal(bs2.coefficients,
                                       np.atleast_2d([-375, -87.5, -75,
                                                      75, -175, 450]))

        bs0 = bspline2.derivative(order=0)
        bs1 = bspline2.derivative()
        bs2 = bspline2.derivative(order=2)

        np.testing.assert_equal(bs1.basis, BSpline(nbasis=4, order=3))
        np.testing.assert_almost_equal(bs1.coefficients,
                                       [[30, -6, -9, -6],
                                        [36, 6, -3, -18],
                                        [12, 0, 0, 12]])
        np.testing.assert_equal(bs0, bspline2)
        np.testing.assert_equal(bs2.basis, BSpline(nbasis=3, order=2))
        np.testing.assert_almost_equal(bs2.coefficients,
                                       [[-144, -6, 12],
                                        [-120, -18, -60],
                                        [-48, 0, 48]])


if __name__ == '__main__':
    print()
    unittest.main()
