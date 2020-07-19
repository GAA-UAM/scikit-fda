from skfda import concatenate
import skfda
from skfda.misc import inner_product, inner_product_matrix
from skfda.representation.basis import (Basis, FDataBasis, Constant, Monomial,
                                        BSpline, Fourier)
from skfda.representation.grid import FDataGrid
import unittest

import numpy as np


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_from_data_cholesky(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), n_basis=5)
        np.testing.assert_array_almost_equal(
            FDataBasis.from_data(x, t, basis, method='cholesky'
                                 ).coefficients.round(2),
            np.array([[1., 2.78, -3., -0.78, 1.]])
        )

    def test_from_data_qr(self):
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSpline((0, 1), n_basis=5)
        np.testing.assert_array_almost_equal(
            FDataBasis.from_data(x, t, basis, method='qr'
                                 ).coefficients.round(2),
            np.array([[1., 2.78, -3., -0.78, 1.]])
        )

    def test_basis_product_generic(self):
        monomial = Monomial(n_basis=5)
        fourier = Fourier(n_basis=3)
        prod = BSpline(n_basis=9, order=8)
        self.assertEqual(Basis.default_basis_of_product(
            monomial, fourier), prod)

    def test_basis_constant_product(self):
        constant = Constant()
        monomial = Monomial()
        fourier = Fourier()
        bspline = BSpline(n_basis=5, order=3)
        self.assertEqual(constant.basis_of_product(monomial), monomial)
        self.assertEqual(constant.basis_of_product(fourier), fourier)
        self.assertEqual(constant.basis_of_product(bspline), bspline)
        self.assertEqual(monomial.basis_of_product(constant), monomial)
        self.assertEqual(fourier.basis_of_product(constant), fourier)
        self.assertEqual(bspline.basis_of_product(constant), bspline)

    def test_basis_fourier_product(self):
        # Test when periods are the same
        fourier = Fourier(n_basis=5)
        fourier2 = Fourier(n_basis=3)
        prod = Fourier(n_basis=7)
        self.assertEqual(fourier.basis_of_product(fourier2), prod)

        # Test when periods are different
        fourier2 = Fourier(n_basis=3, period=2)
        prod = BSpline(n_basis=9, order=8)
        self.assertEqual(fourier.basis_of_product(fourier2), prod)

    def test_basis_monomial_product(self):
        monomial = Monomial(n_basis=5)
        monomial2 = Monomial(n_basis=3)
        prod = Monomial(n_basis=8)
        self.assertEqual(monomial.basis_of_product(monomial2), prod)

    def test_basis_bspline_product(self):
        bspline = BSpline(n_basis=6, order=4)
        bspline2 = BSpline(domain_range=(0, 1), n_basis=6,
                           order=4, knots=[0, 0.3, 1 / 3, 1])
        prod = BSpline(domain_range=(0, 1), n_basis=10, order=7,
                       knots=[0, 0.3, 1 / 3, 2 / 3, 1])
        self.assertEqual(bspline.basis_of_product(bspline2), prod)

    def test_basis_inner_matrix(self):
        np.testing.assert_array_almost_equal(
            Monomial(n_basis=3).inner_product_matrix(),
            [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4], [1 / 3, 1 / 4, 1 / 5]])

        np.testing.assert_array_almost_equal(
            Monomial(n_basis=3).inner_product_matrix(Monomial(n_basis=3)),
            [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4], [1 / 3, 1 / 4, 1 / 5]])

        np.testing.assert_array_almost_equal(
            Monomial(n_basis=3).inner_product_matrix(Monomial(n_basis=4)),
            [[1, 1 / 2, 1 / 3, 1 / 4],
             [1 / 2, 1 / 3, 1 / 4, 1 / 5],
             [1 / 3, 1 / 4, 1 / 5, 1 / 6]])

        # TODO testing with other basis

    def test_basis_gram_matrix_monomial(self):

        basis = Monomial(n_basis=3)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()
        gram_matrix_res = np.array([[1, 1 / 2, 1 / 3],
                                    [1 / 2, 1 / 3, 1 / 4],
                                    [1 / 3, 1 / 4, 1 / 5]])

        np.testing.assert_allclose(
            gram_matrix, gram_matrix_res)
        np.testing.assert_allclose(
            gram_matrix_numerical, gram_matrix_res)

    def test_basis_gram_matrix_fourier(self):

        basis = Fourier(n_basis=3)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()
        gram_matrix_res = np.identity(3)

        np.testing.assert_allclose(
            gram_matrix, gram_matrix_res)
        np.testing.assert_allclose(
            gram_matrix_numerical, gram_matrix_res, atol=1e-15, rtol=1e-15)

    def test_basis_gram_matrix_bspline(self):

        basis = BSpline(n_basis=6)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()
        gram_matrix_res = np.array(
            [[0.04761905, 0.02916667, 0.00615079,
              0.00039683, 0., 0.],
             [0.02916667, 0.07380952, 0.05208333,
              0.01145833, 0.00014881, 0.],
             [0.00615079, 0.05208333, 0.10892857, 0.07098214,
              0.01145833, 0.00039683],
             [0.00039683, 0.01145833, 0.07098214, 0.10892857,
              0.05208333, 0.00615079],
             [0., 0.00014881, 0.01145833, 0.05208333,
              0.07380952, 0.02916667],
             [0., 0., 0.00039683, 0.00615079,
              0.02916667, 0.04761905]])

        np.testing.assert_allclose(
            gram_matrix, gram_matrix_res, rtol=1e-4)
        np.testing.assert_allclose(
            gram_matrix_numerical, gram_matrix_res, rtol=1e-4)

    def test_basis_basis_inprod(self):
        monomial = Monomial(n_basis=4)
        bspline = BSpline(n_basis=5, order=4)
        np.testing.assert_allclose(
            monomial.inner_product_matrix(bspline),
            np.array(
                [[0.12499983, 0.25000035, 0.24999965, 0.25000035, 0.12499983],
                 [0.01249991, 0.07500017, 0.12499983, 0.17500017, 0.11249991],
                 [0.00208338, 0.02916658, 0.07083342, 0.12916658, 0.10208338],
                 [0.00044654, 0.01339264, 0.04375022, 0.09910693, 0.09330368]
                 ]), rtol=1e-3)
        np.testing.assert_array_almost_equal(
            monomial.inner_product_matrix(bspline),
            bspline.inner_product_matrix(monomial).T
        )

    def test_basis_fdatabasis_inprod(self):
        monomial = Monomial(n_basis=4)
        bspline = BSpline(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(monomial, bsplinefd),
            np.array([[2., 7., 12.],
                      [1.29626206, 3.79626206, 6.29626206],
                      [0.96292873, 2.62959539, 4.29626206],
                      [0.7682873, 2.0182873, 3.2682873]]), rtol=1e-4)

    def test_fdatabasis_fdatabasis_inprod(self):
        monomial = Monomial(n_basis=4)
        monomialfd = FDataBasis(monomial, [[5, 4, 1, 0],
                                           [4, 2, 1, 0],
                                           [4, 1, 6, 4],
                                           [4, 5, 0, 1],
                                           [5, 6, 2, 0]])
        bspline = BSpline(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(monomialfd, bsplinefd),
            np.array([[16.14797697, 52.81464364, 89.4813103],
                      [11.55565285, 38.22211951, 64.88878618],
                      [18.14698361, 55.64698361, 93.14698361],
                      [15.2495976, 48.9995976, 82.7495976],
                      [19.70392982, 63.03676315, 106.37009648]]),
            rtol=1e-4)

    def test_comutativity_inprod(self):
        monomial = Monomial(n_basis=4)
        bspline = BSpline(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(bsplinefd, monomial),
            np.transpose(inner_product_matrix(monomial, bsplinefd))
        )

    def test_fdatabasis_times_fdatabasis_fdatabasis(self):
        monomial = FDataBasis(Monomial(n_basis=3), [1, 2, 3])
        bspline = FDataBasis(BSpline(n_basis=6, order=4), [1, 2, 4, 1, 0, 1])
        times_fdar = monomial.times(bspline)

        prod_basis = BSpline(n_basis=9, order=6, knots=[0, 0.25, 0.5, 0.75, 1])
        prod_coefs = np.array([[0.9788352,  1.6289955,  2.7004969,  6.2678739,
                                8.7636441,  4.0069960,  0.7126961,  2.8826708,
                                6.0052311]])

        self.assertEqual(prod_basis, times_fdar.basis)
        np.testing.assert_array_almost_equal(
            prod_coefs, times_fdar.coefficients)

    def test_fdatabasis_times_fdatabasis_list(self):
        monomial = FDataBasis(Monomial(n_basis=3),
                              [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = monomial.times([3, 2, 1])

        expec_basis = Monomial(n_basis=3)
        expec_coefs = np.array([[3, 6, 9], [8, 10, 12], [7, 8, 9]])

        self.assertEqual(expec_basis, result.basis)
        np.testing.assert_array_almost_equal(expec_coefs, result.coefficients)

    def test_fdatabasis_times_fdatabasis_int(self):
        monomial = FDataBasis(Monomial(n_basis=3),
                              [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = monomial.times(3)

        expec_basis = Monomial(n_basis=3)
        expec_coefs = np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]])

        self.assertEqual(expec_basis, result.basis)
        np.testing.assert_array_almost_equal(expec_coefs, result.coefficients)

    def test_fdatabasis__add__(self):
        monomial1 = FDataBasis(Monomial(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(Monomial(n_basis=3), [[1, 2, 3], [3, 4, 5]])

        np.testing.assert_equal(monomial1 + monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 4, 6], [4, 6, 8]]))
        np.testing.assert_equal(monomial2 + 1,
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 2, 3], [4, 4, 5]]))
        np.testing.assert_equal(1 + monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 2, 3], [4, 4, 5]]))
        np.testing.assert_equal(monomial2 + [1, 2],
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 2, 3], [5, 4, 5]]))
        np.testing.assert_equal([1, 2] + monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 2, 3], [5, 4, 5]]))

        with np.testing.assert_raises(TypeError):
            monomial2 + FDataBasis(Fourier(n_basis=3),
                                   [[2, 2, 3], [5, 4, 5]])

    def test_fdatabasis__sub__(self):
        monomial1 = FDataBasis(Monomial(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(Monomial(n_basis=3), [[1, 2, 3], [3, 4, 5]])

        np.testing.assert_equal(monomial1 - monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[0, 0, 0], [-2, -2, -2]]))
        np.testing.assert_equal(monomial2 - 1,
                                FDataBasis(Monomial(n_basis=3),
                                           [[0, 2, 3], [2, 4, 5]]))
        np.testing.assert_equal(1 - monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[0, -2, -3], [-2, -4, -5]]))
        np.testing.assert_equal(monomial2 - [1, 2],
                                FDataBasis(Monomial(n_basis=3),
                                           [[0, 2, 3], [1, 4, 5]]))
        np.testing.assert_equal([1, 2] - monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[0, -2, -3], [-1, -4, -5]]))

        with np.testing.assert_raises(TypeError):
            monomial2 - FDataBasis(Fourier(n_basis=3),
                                   [[2, 2, 3], [5, 4, 5]])

    def test_fdatabasis__mul__(self):
        monomial1 = FDataBasis(Monomial(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(Monomial(n_basis=3), [[1, 2, 3], [3, 4, 5]])

        np.testing.assert_equal(monomial1 * 2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[2, 4, 6]]))
        np.testing.assert_equal(3 * monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[3, 6, 9], [9, 12, 15]]))
        np.testing.assert_equal(3 * monomial2,
                                monomial2 * 3)

        np.testing.assert_equal(monomial2 * [1, 2],
                                FDataBasis(Monomial(n_basis=3),
                                           [[1, 2, 3], [6, 8, 10]]))
        np.testing.assert_equal([1, 2] * monomial2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[1, 2, 3], [6, 8, 10]]))

        with np.testing.assert_raises(TypeError):
            monomial2 * FDataBasis(Fourier(n_basis=3),
                                   [[2, 2, 3], [5, 4, 5]])

        with np.testing.assert_raises(TypeError):
            monomial2 * monomial2

    def test_fdatabasis__mul__2(self):
        monomial1 = FDataBasis(Monomial(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(Monomial(n_basis=3), [[1, 2, 3], [3, 4, 5]])

        np.testing.assert_equal(monomial1 / 2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[1 / 2, 1, 3 / 2]]))
        np.testing.assert_equal(monomial2 / 2,
                                FDataBasis(Monomial(n_basis=3),
                                           [[1 / 2, 1, 3 / 2], [3 / 2, 2, 5 / 2]]))

        np.testing.assert_equal(monomial2 / [1, 2],
                                FDataBasis(Monomial(n_basis=3),
                                           [[1, 2, 3], [3 / 2, 2, 5 / 2]]))

    def test_fdatabasis_derivative_constant(self):
        monomial = FDataBasis(Monomial(n_basis=8),
                              [1, 5, 8, 9, 7, 8, 4, 5])
        monomial2 = FDataBasis(Monomial(n_basis=5),
                               [[4, 9, 7, 4, 3],
                                [1, 7, 9, 8, 5],
                                [4, 6, 6, 6, 8]])

        np.testing.assert_equal(monomial.derivative(),
                                FDataBasis(Monomial(n_basis=7),
                                           [5, 16, 27, 28, 40, 24, 35]))
        np.testing.assert_equal(monomial.derivative(order=0), monomial)
        np.testing.assert_equal(monomial.derivative(order=6),
                                FDataBasis(Monomial(n_basis=2),
                                           [2880, 25200]))

        np.testing.assert_equal(monomial2.derivative(),
                                FDataBasis(Monomial(n_basis=4),
                                           [[9, 14, 12, 12],
                                            [7, 18, 24, 20],
                                            [6, 12, 18, 32]]))
        np.testing.assert_equal(monomial2.derivative(order=0), monomial2)
        np.testing.assert_equal(monomial2.derivative(order=3),
                                FDataBasis(Monomial(n_basis=2),
                                           [[24, 72],
                                            [48, 120],
                                            [36, 192]]))

    def test_fdatabasis_derivative_monomial(self):
        monomial = FDataBasis(Monomial(n_basis=8),
                              [1, 5, 8, 9, 7, 8, 4, 5])
        monomial2 = FDataBasis(Monomial(n_basis=5),
                               [[4, 9, 7, 4, 3],
                                [1, 7, 9, 8, 5],
                                [4, 6, 6, 6, 8]])

        np.testing.assert_equal(monomial.derivative(),
                                FDataBasis(Monomial(n_basis=7),
                                           [5, 16, 27, 28, 40, 24, 35]))
        np.testing.assert_equal(monomial.derivative(order=0), monomial)
        np.testing.assert_equal(monomial.derivative(order=6),
                                FDataBasis(Monomial(n_basis=2),
                                           [2880, 25200]))

        np.testing.assert_equal(monomial2.derivative(),
                                FDataBasis(Monomial(n_basis=4),
                                           [[9, 14, 12, 12],
                                            [7, 18, 24, 20],
                                            [6, 12, 18, 32]]))
        np.testing.assert_equal(monomial2.derivative(order=0), monomial2)
        np.testing.assert_equal(monomial2.derivative(order=3),
                                FDataBasis(Monomial(n_basis=2),
                                           [[24, 72],
                                            [48, 120],
                                            [36, 192]]))

    def test_fdatabasis_derivative_fourier(self):
        fourier = FDataBasis(Fourier(n_basis=7),
                             [1, 5, 8, 9, 8, 4, 5])
        fourier2 = FDataBasis(Fourier(n_basis=5),
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
                                        [0, -56.54867, 43.98230, -
                                            62.83185, 100.53096],
                                        [0, -37.69911, 37.69911, -100.53096, 75.39822]])
        np.testing.assert_equal(fou0, fourier2)
        np.testing.assert_equal(fou2.basis, fourier2.basis)
        np.testing.assert_almost_equal(fou2.coefficients.round(5),
                                       [[0, -355.30576, -276.34892, -631.65468, -473.74101],
                                        [0, -276.34892, -355.30576, -
                                            1263.30936, -789.56835],
                                        [0, -236.87051, -236.87051, -947.48202, -1263.30936]])

    def test_fdatabasis_derivative_bspline(self):
        bspline = FDataBasis(BSpline(n_basis=8),
                             [1, 5, 8, 9, 7, 8, 4, 5])
        bspline2 = FDataBasis(BSpline(n_basis=5),
                              [[4, 9, 7, 4, 3],
                               [1, 7, 9, 8, 5],
                               [4, 6, 6, 6, 8]])

        bs0 = bspline.derivative(order=0)
        bs1 = bspline.derivative()
        bs2 = bspline.derivative(order=2)
        np.testing.assert_equal(bs1.basis, BSpline(n_basis=7, order=3))
        np.testing.assert_almost_equal(bs1.coefficients,
                                       np.atleast_2d([60, 22.5, 5,
                                                      -10, 5, -30, 15]))
        np.testing.assert_equal(bs0, bspline)
        np.testing.assert_equal(bs2.basis, BSpline(n_basis=6, order=2))
        np.testing.assert_almost_equal(bs2.coefficients,
                                       np.atleast_2d([-375, -87.5, -75,
                                                      75, -175, 450]))

        bs0 = bspline2.derivative(order=0)
        bs1 = bspline2.derivative()
        bs2 = bspline2.derivative(order=2)

        np.testing.assert_equal(bs1.basis, BSpline(n_basis=4, order=3))
        np.testing.assert_almost_equal(bs1.coefficients,
                                       [[30, -6, -9, -6],
                                        [36, 6, -3, -18],
                                        [12, 0, 0, 12]])
        np.testing.assert_equal(bs0, bspline2)
        np.testing.assert_equal(bs2.basis, BSpline(n_basis=3, order=2))
        np.testing.assert_almost_equal(bs2.coefficients,
                                       [[-144, -6, 12],
                                        [-120, -18, -60],
                                        [-48, 0, 48]])

    def test_concatenate(self):
        sample1 = np.arange(0, 10)
        sample2 = np.arange(10, 20)
        fd1 = FDataGrid([sample1]).to_basis(Fourier(n_basis=5))
        fd2 = FDataGrid([sample2]).to_basis(Fourier(n_basis=5))

        fd = concatenate([fd1, fd2])

        np.testing.assert_equal(fd.n_samples, 2)
        np.testing.assert_equal(fd.dim_codomain, 1)
        np.testing.assert_equal(fd.dim_domain, 1)
        np.testing.assert_array_equal(fd.coefficients, np.concatenate(
            [fd1.coefficients, fd2.coefficients]))

    def test_vector_valued(self):
        X, y = skfda.datasets.fetch_weather(return_X_y=True)

        basis_dim = skfda.representation.basis.Fourier(
            n_basis=7, domain_range=X.domain_range)
        basis = skfda.representation.basis.VectorValued(
            [basis_dim] * 2
        )

        X_basis = X.to_basis(basis)

        self.assertEqual(X_basis.dim_codomain, 2)

        self.assertEqual(X_basis.coordinates[0].basis, basis_dim)
        np.testing.assert_allclose(
            X_basis.coordinates[0].coefficients,
            X.coordinates[0].to_basis(basis_dim).coefficients)

        self.assertEqual(X_basis.coordinates[1].basis, basis_dim)
        np.testing.assert_allclose(
            X_basis.coordinates[1].coefficients,
            X.coordinates[1].to_basis(basis_dim).coefficients)


if __name__ == '__main__':
    print()
    unittest.main()
