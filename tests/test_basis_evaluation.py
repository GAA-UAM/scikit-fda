
from skfda.representation.basis import (
    FDataBasis, Monomial, BSpline, Fourier, Constant, VectorValued, Tensor)
import unittest

import numpy as np


class TestBasisEvaluationFourier(unittest.TestCase):

    def test_evaluation_simple_fourier(self):
        """Test the evaluation of FDataBasis"""
        fourier = Fourier(domain_range=(0, 2), n_basis=5)

        coefficients = np.array([[1,  2,  3,  4,  5],
                                 [6,  7,  8,  9, 10]])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array([[8.71,  9.66,  1.84, -4.71, -2.80, 2.71,
                         2.45, -3.82,  -6.66, -0.30,  8.71],
                        [22.24, 26.48, 10.57, -4.95, -3.58, 6.24,
                         5.31, -7.69, -13.32,  1.13, 22.24]])[..., np.newaxis]

        np.testing.assert_array_almost_equal(f(t).round(2), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(2), res)

    def test_evaluation_point_fourier(self):
        """Test the evaluation of a single point FDataBasis"""
        fourier = Fourier(domain_range=(0, 1), n_basis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)

        # Test different ways of call f with a point
        res = np.array([-0.903918107989282, -0.267163981229459]
                       ).reshape((2, 1, 1)).round(4)

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_fourier(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        fourier = Fourier(domain_range=(0, 1), n_basis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([4.34138447771721, -7.09352774867064, 2.75214327095343,
                        4.34138447771721, 6.52573053999253,
                        -4.81336320468984, -1.7123673353027, 6.52573053999253]
                       ).reshape((2, 4, 1)).round(3)

        f_deriv = f.derivative()
        np.testing.assert_array_almost_equal(
            f_deriv(t).round(3), res
        )

    def test_evaluation_grid_fourier(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        fourier = Fourier(domain_range=(0, 1), n_basis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Different ways to pass the axes
        np.testing.assert_array_almost_equal(f(t, grid=True), res_test)
        np.testing.assert_array_almost_equal(f((t,), grid=True), res_test)
        np.testing.assert_array_almost_equal(f([t], grid=True), res_test)
        np.testing.assert_array_almost_equal(f(np.atleast_2d(t), grid=True),
                                             res_test)

        # Number of axis different than the domain dimension (1)
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_composed_fourier(self):
        """Test the evaluation of FDataBasis the a matrix of times instead of
        a list of times """
        fourier = Fourier(domain_range=(0, 1), n_basis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        t = np.linspace(0, 1, 4)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]),
                                             f([[1], [1]],
                                               aligned=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned=False)[1])

    def test_domain_in_list_fourier(self):
        """Test the evaluation of FDataBasis"""
        for fourier in (Fourier(domain_range=[(0, 1)], n_basis=3),
                        Fourier(domain_range=((0, 1),), n_basis=3),
                        Fourier(domain_range=np.array((0, 1)), n_basis=3),
                        Fourier(domain_range=np.array([(0, 1)]), n_basis=3)):

            coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                     [0.01778079, 0.73440271, 0.20148638]])

            f = FDataBasis(fourier, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([0.905, 0.147, -1.05, 0.905, 0.303,
                            0.775, -1.024, 0.303]).reshape((2, 4, 1))

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)


class TestBasisEvaluationBSpline(unittest.TestCase):

    def test_evaluation_simple_bspline(self):
        """Test the evaluation of FDataBasis"""
        bspline = BSpline(domain_range=(0, 2), n_basis=5)

        coefficients = np.array([[1,  2,  3,  4,  5],
                                 [6,  7,  8,  9, 10]])

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array([[1, 1.54, 1.99, 2.37, 2.7, 3,
                         3.3, 3.63, 4.01, 4.46, 5],
                        [6, 6.54, 6.99, 7.37, 7.7, 8,
                         8.3, 8.63, 9.01, 9.46, 10]])[..., np.newaxis]

        np.testing.assert_array_almost_equal(f(t).round(2), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(2), res)

    def test_evaluation_point_bspline(self):
        """Test the evaluation of a single point FDataBasis"""
        bspline = BSpline(domain_range=(0, 1), n_basis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)

        # Test different ways of call f with a point
        res = np.array([[0.5696], [0.3104]])[..., np.newaxis]

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_bspline(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        bspline = BSpline(domain_range=(0, 1), n_basis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 1, 4)

        f_deriv = f.derivative()
        np.testing.assert_array_almost_equal(
            f_deriv(t).round(3),
            np.array([[2.927,  0.453, -1.229,  0.6],
                      [4.3, -1.599,  1.016, -2.52]])[..., np.newaxis]
        )

    def test_evaluation_grid_bspline(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        bspline = BSpline(domain_range=(0, 1), n_basis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Different ways to pass the axes
        np.testing.assert_array_almost_equal(f(t, grid=True), res_test)
        np.testing.assert_array_almost_equal(f((t,), grid=True), res_test)
        np.testing.assert_array_almost_equal(f([t], grid=True), res_test)
        np.testing.assert_array_almost_equal(
            f(np.atleast_2d(t), grid=True), res_test)

        # Number of axis different than the domain dimension (1)
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_composed_bspline(self):
        """Test the evaluation of FDataBasis the a matrix of times instead of
        a list of times """
        bspline = BSpline(domain_range=(0, 1), n_basis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        t = np.linspace(0, 1, 4)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]),
                                             f([[1], [1]],
                                               aligned=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned=False)[1])

    def test_domain_in_list_bspline(self):
        """Test the evaluation of FDataBasis"""

        for bspline in (BSpline(domain_range=[(0, 1)], n_basis=5, order=3),
                        BSpline(domain_range=((0, 1),), n_basis=5, order=3),
                        BSpline(domain_range=np.array((0, 1)), n_basis=5,
                                order=3),
                        BSpline(domain_range=np.array([(0, 1)]), n_basis=5,
                                order=3)
                        ):

            coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                            [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

            f = FDataBasis(bspline, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([[0.001, 0.564, 0.435, 0.33],
                            [0.018, 0.468, 0.371, 0.12]])[..., np.newaxis]

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)

        # Check error
        with np.testing.assert_raises(ValueError):
            BSpline(domain_range=[(0, 1), (0, 1)])


class TestBasisEvaluationMonomial(unittest.TestCase):

    def test_evaluation_simple_monomial(self):
        """Test the evaluation of FDataBasis"""

        monomial = Monomial(domain_range=(0, 2), n_basis=5)

        coefficients = np.array([[1,  2,  3,  4,  5],
                                 [6,  7,  8,  9, 10]])

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array(
            [[1.00, 1.56, 2.66, 4.79, 8.62, 15.00,
              25.00, 39.86, 61.03, 90.14, 129.00],
             [6.00, 7.81, 10.91, 16.32, 25.42, 40.00,
              62.21, 94.59, 140.08, 201.98, 284.00]])[..., np.newaxis]

        np.testing.assert_array_almost_equal(f(t).round(2), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(2), res)

    def test_evaluation_point_monomial(self):
        """Test the evaluation of a single point FDataBasis"""
        monomial = Monomial(domain_range=(0, 1), n_basis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)

        # Test different ways of call f with a point
        res = np.array([[2.75], [1.525]])[..., np.newaxis]

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_monomial(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        monomial = Monomial(domain_range=(0, 1), n_basis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 1, 4)

        f_deriv = f.derivative()
        np.testing.assert_array_almost_equal(
            f_deriv(t).round(3),
            np.array([[2., 4., 6., 8.],
                      [1.4, 2.267, 3.133, 4.]])[..., np.newaxis]
        )

    def test_evaluation_grid_monomial(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        monomial = Monomial(domain_range=(0, 1), n_basis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Different ways to pass the axes
        np.testing.assert_array_almost_equal(f(t, grid=True), res_test)
        np.testing.assert_array_almost_equal(f((t,), grid=True), res_test)
        np.testing.assert_array_almost_equal(f([t], grid=True), res_test)
        np.testing.assert_array_almost_equal(
            f(np.atleast_2d(t), grid=True), res_test)

        # Number of axis different than the domain dimension (1)
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_composed_monomial(self):
        """Test the evaluation of FDataBasis the a matrix of times instead of
        a list of times """
        monomial = Monomial(domain_range=(0, 1), n_basis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        t = np.linspace(0, 1, 4)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]),
                                             f([[1], [1]],
                                               aligned=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned=False)[1])

    def test_domain_in_list_monomial(self):
        """Test the evaluation of FDataBasis"""

        for monomial in (Monomial(domain_range=[(0, 1)], n_basis=3),
                         Monomial(domain_range=((0, 1),), n_basis=3),
                         Monomial(domain_range=np.array((0, 1)), n_basis=3),
                         Monomial(domain_range=np.array([(0, 1)]), n_basis=3)):

            coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

            f = FDataBasis(monomial, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([[1., 2., 3.667, 6.],
                            [0.5, 1.111, 2.011, 3.2]])[..., np.newaxis]

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)


class TestBasisEvaluationVectorValued(unittest.TestCase):

    def test_vector_valued_constant(self):

        basis_first = Constant()
        basis_second = Constant()

        basis = VectorValued([basis_first, basis_second])

        fd = FDataBasis(basis=basis, coefficients=[[1, 2], [3, 4]])

        self.assertEqual(fd.dim_codomain, 2)

        res = np.array([[[1, 2]], [[3, 4]]])

        np.testing.assert_allclose(fd(0), res)

    def test_vector_valued_constant_monomial(self):

        basis_first = Constant(domain_range=(0, 5))
        basis_second = Monomial(n_basis=3, domain_range=(0, 5))

        basis = VectorValued([basis_first, basis_second])

        fd = FDataBasis(basis=basis, coefficients=[
                        [1, 2, 3, 4], [3, 4, 5, 6]])

        self.assertEqual(fd.dim_codomain, 2)

        np.testing.assert_allclose(fd.domain_range[0], (0, 5))

        res = np.array([[[1, 2], [1, 9], [1, 24]],
                        [[3, 4], [3, 15], [3, 38]]])

        np.testing.assert_allclose(fd([0, 1, 2]), res)


class TestBasisEvaluationTensor(unittest.TestCase):

    def test_tensor_monomial_constant(self):

        basis = Tensor([Monomial(n_basis=2), Constant()])

        fd = FDataBasis(basis=basis, coefficients=[1, 1])

        self.assertEqual(fd.dim_domain, 2)
        self.assertEqual(fd.dim_codomain, 1)

        np.testing.assert_allclose(fd([0., 0.]), [[[1.]]])

        np.testing.assert_allclose(fd([0.5, 0.5]), [[[1.5]]])

        np.testing.assert_allclose(
            fd([(0., 0.), (0.5, 0.5)]), [[[1.0], [1.5]]])

        fd_grid = fd.to_grid()

        fd2 = fd_grid.to_basis(basis)

        np.testing.assert_allclose(fd.coefficients, fd2.coefficients)


if __name__ == '__main__':
    print()
    unittest.main()
