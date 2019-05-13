
import unittest
from skfda.representation.basis import FDataBasis, Monomial, BSpline, Fourier
import numpy as np


class TestBasisEvaluationFourier(unittest.TestCase):

    def test_evaluation_simple_fourier(self):
        """Test the evaluation of FDataBasis"""
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([0.905482867989282, 0.146814813180645, -1.04995054116993,
                        0.905482867989282, 0.302725561229459,
                        0.774764356993855, -1.02414754822331, 0.302725561229459]
                       ).reshape((2, 4)).round(3)

        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)

    def test_evaluation_point_fourier(self):
        """Test the evaluation of a single point FDataBasis"""
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)

        # Test different ways of call f with a point
        res = np.array([-0.903918107989282, -0.267163981229459]
                       ).reshape((2, 1)).round(4)

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_fourier(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([4.34138447771721, -7.09352774867064, 2.75214327095343,
                        4.34138447771721, 6.52573053999253,
                        -4.81336320468984, -1.7123673353027, 6.52573053999253]
                       ).reshape((2, 4)).round(3)

        np.testing.assert_array_almost_equal(
            f(t, derivative=1).round(3), res
        )

    def test_evaluation_grid_fourier(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

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
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]), f([[1], [1]],
                                                       aligned_evaluation=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned_evaluation=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned_evaluation=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned_evaluation=False)[1])

    def test_evaluation_keepdims_fourier(self):
        """Test behaviour of keepdims """
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        f_keepdims = FDataBasis(fourier, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([0.905482867989282, 0.146814813180645, -1.04995054116993,
                        0.905482867989282, 0.302725561229459,
                        0.774764356993855, -1.02414754822331, 0.302725561229459]
                       ).reshape((2, 4)).round(3)

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(
            f(t, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(
            f_keepdims(t).round(3), res_keepdims)
        np.testing.assert_array_almost_equal(f_keepdims(t, keepdims=False
                                                        ).round(3),
                                             res)
        np.testing.assert_array_almost_equal(f_keepdims(t, keepdims=True
                                                        ).round(3),
                                             res_keepdims)

    def test_evaluation_composed_keepdims_fourier(self):
        """Test behaviour of keepdims with composed evaluation"""
        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        f_keepdims = FDataBasis(fourier, coefficients, keepdims=True)

        t = [[0, 0.5, 0.6], [0.2, 0.7, 0.1]]

        res = np.array([[0.69173518, -0.69017042, -1.08997978],
                        [0.60972512, -0.57416354,  1.02551401]]).round(3)

        res = np.array([0.905482867989282, -0.903918107989282,
                        -1.13726755517372, 1.09360302608278,
                        -1.05804144608278, 0.85878105128844]
                       ).reshape((2, 3)).round(3)

        res_keepdims = res.reshape((2, 3, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False
                                               ).round(3),
                                             res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(f_keepdims(t,
                                                        aligned_evaluation=False
                                                        ).round(3),
                                             res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=False).round(3),
            res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=True).round(3),
            res_keepdims)

    def test_evaluation_grid_keepdims_fourier(self):
        """Test behaviour of keepdims with grid evaluation"""

        fourier = Fourier(domain_range=(0, 1), nbasis=3)

        coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                 [0.01778079, 0.73440271, 0.20148638]])

        f = FDataBasis(fourier, coefficients)
        f_keepdims = FDataBasis(fourier, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([0.905482867989282, 0.146814813180645, -1.04995054116993,
                        0.905482867989282, 0.302725561229459,
                        0.774764356993855, -1.02414754822331, 0.302725561229459]
                       ).reshape((2, 4)).round(3)

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t, grid=True).round(3), res)
        np.testing.assert_array_almost_equal(f(t, grid=True, keepdims=False
                                               ).round(3),
                                             res)

        np.testing.assert_array_almost_equal(f(t,  grid=True, keepdims=True
                                               ).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(f_keepdims(t, grid=True
                                                        ).round(3),
                                             res_keepdims)
        np.testing.assert_array_almost_equal(f_keepdims(t, grid=True,
                                                        keepdims=False
                                                        ).round(3), res)
        np.testing.assert_array_almost_equal(f_keepdims(t, grid=True,
                                                        keepdims=True).round(3),
                                             res_keepdims)

    def test_domain_in_list_fourier(self):
        """Test the evaluation of FDataBasis"""
        for fourier in (Fourier(domain_range=[(0, 1)], nbasis=3),
                        Fourier(domain_range=((0, 1),), nbasis=3),
                        Fourier(domain_range=np.array((0, 1)), nbasis=3),
                        Fourier(domain_range=np.array([(0, 1)]), nbasis=3)):

            coefficients = np.array([[0.00078238, 0.48857741, 0.63971985],
                                     [0.01778079, 0.73440271, 0.20148638]])

            f = FDataBasis(fourier, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([0.905, 0.147, -1.05, 0.905, 0.303,
                            0.775, -1.024, 0.303]).reshape((2, 4))

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)


class TestBasisEvaluationBSpline(unittest.TestCase):

    def test_evaluation_simple_bspline(self):
        """Test the evaluation of FDataBasis"""
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([[0.001, 0.564, 0.435, 0.33],
                        [0.018, 0.468, 0.371, 0.12]])

        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)

    def test_evaluation_point_bspline(self):
        """Test the evaluation of a single point FDataBasis"""
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)

        # Test different ways of call f with a point
        res = np.array([[0.5696], [0.3104]])

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_bspline(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 1, 4)

        np.testing.assert_array_almost_equal(
            f(t, derivative=1).round(3),
            np.array([[2.927,  0.453, -1.229,  0.6],
                      [4.3, -1.599,  1.016, -2.52]])
        )

    def test_evaluation_grid_bspline(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

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
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]),
                                             f([[1], [1]],
                                               aligned_evaluation=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned_evaluation=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned_evaluation=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned_evaluation=False)[1])

    def test_evaluation_keepdims_bspline(self):
        """Test behaviour of keepdims """
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        f_keepdims = FDataBasis(bspline, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([[0.001, 0.564, 0.435, 0.33],
                        [0.018, 0.468, 0.371, 0.12]])

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(
            f(t, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(
            f_keepdims(t).round(3), res_keepdims)
        np.testing.assert_array_almost_equal(f_keepdims(t, keepdims=False
                                                        ).round(3),
                                             res)
        np.testing.assert_array_almost_equal(f_keepdims(t, keepdims=True
                                                        ).round(3),
                                             res_keepdims)

    def test_evaluation_composed_keepdims_bspline(self):
        """Test behaviour of keepdims with composed evaluation"""
        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        f_keepdims = FDataBasis(bspline, coefficients, keepdims=True)

        t = [[0, 0.5, 0.6], [0.2, 0.7, 0.1]]

        res = np.array([[0.001, 0.57, 0.506],
                        [0.524, 0.399, 0.359]])

        res_keepdims = res.reshape((2, 3, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False
                                               ).round(3),
                                             res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=False).round(3),
                                             res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(f_keepdims(t,
                                                        aligned_evaluation=False
                                                        ).round(3),
                                             res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=False).round(3),
            res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=True).round(3),
            res_keepdims)

    def test_evaluation_grid_keepdims_bspline(self):
        """Test behaviour of keepdims with grid evaluation"""

        bspline = BSpline(domain_range=(0, 1), nbasis=5, order=3)

        coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                        [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

        f = FDataBasis(bspline, coefficients)
        f_keepdims = FDataBasis(bspline, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([[0.001, 0.564, 0.435, 0.33],
                        [0.018, 0.468, 0.371, 0.12]])

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t, grid=True).round(3), res)
        np.testing.assert_array_almost_equal(
            f(t, grid=True, keepdims=False).round(3), res)

        np.testing.assert_array_almost_equal(
            f(t,  grid=True, keepdims=True).round(3),
            res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(f_keepdims(t, grid=True).round(3),
                                             res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, grid=True, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, grid=True, keepdims=True).round(3),
            res_keepdims)

    def test_domain_in_list_bspline(self):
        """Test the evaluation of FDataBasis"""

        for bspline in (BSpline(domain_range=[(0, 1)], nbasis=5, order=3),
                        BSpline(domain_range=((0, 1),), nbasis=5, order=3),
                        BSpline(domain_range=np.array((0, 1)), nbasis=5,
                                order=3),
                        BSpline(domain_range=np.array([(0, 1)]), nbasis=5,
                                order=3)
                        ):

            coefficients = [[0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
                            [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12]]

            f = FDataBasis(bspline, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([[0.001, 0.564, 0.435, 0.33],
                            [0.018, 0.468, 0.371, 0.12]])

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)

        # Check error
        with np.testing.assert_raises(ValueError):
            BSpline(domain_range=[(0, 1), (0, 1)])


class TestBasisEvaluationMonomial(unittest.TestCase):

    def test_evaluation_simple_monomial(self):
        """Test the evaluation of FDataBasis"""

        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([[1., 2., 3.667, 6.],
                        [0.5, 1.111, 2.011, 3.2]])

        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)

    def test_evaluation_point_monomial(self):
        """Test the evaluation of a single point FDataBasis"""
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)

        # Test different ways of call f with a point
        res = np.array([[2.75], [1.525]])

        np.testing.assert_array_almost_equal(f([0.5]).round(4), res)
        np.testing.assert_array_almost_equal(f((0.5,)).round(4), res)
        np.testing.assert_array_almost_equal(f(0.5).round(4), res)
        np.testing.assert_array_almost_equal(f(np.array([0.5])).round(4), res)

        # Problematic case, should be accepted or no?
        #np.testing.assert_array_almost_equal(f(np.array(0.5)).round(4), res)

    def test_evaluation_derivative_monomial(self):
        """Test the evaluation of the derivative of a FDataBasis"""
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 1, 4)

        np.testing.assert_array_almost_equal(
            f(t, derivative=1).round(3),
            np.array([[2., 4., 6., 8.],
                      [1.4, 2.267, 3.133, 4.]])
        )

    def test_evaluation_grid_monomial(self):
        """Test the evaluation of FDataBasis with the grid option set to
            true. Nothing should be change due to the domain dimension is 1,
            but can accept the """
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

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
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Test same result than evaluation standart
        np.testing.assert_array_almost_equal(f([1]), f([[1], [1]],
                                                       aligned_evaluation=False))
        np.testing.assert_array_almost_equal(f(t), f(np.vstack((t, t)),
                                                     aligned_evaluation=False))

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_array_almost_equal(f(t_multiple[0])[0],
                                             f(t_multiple,
                                               aligned_evaluation=False)[0])
        np.testing.assert_array_almost_equal(f(t_multiple[1])[1],
                                             f(t_multiple,
                                               aligned_evaluation=False)[1])

    def test_evaluation_keepdims_monomial(self):
        """Test behaviour of keepdims """
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        f_keepdims = FDataBasis(monomial, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([[1., 2., 3.667, 6.],
                        [0.5, 1.111, 2.011, 3.2]])

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t).round(3), res)
        np.testing.assert_array_almost_equal(
            f(t, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(
            f_keepdims(t).round(3), res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, keepdims=True).round(3), res_keepdims)

    def test_evaluation_composed_keepdims_monomial(self):
        """Test behaviour of keepdims with composed evaluation"""
        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        f_keepdims = FDataBasis(monomial, coefficients, keepdims=True)

        t = [[0, 0.5, 0.6], [0.2, 0.7, 0.1]]

        res = np.array([[1., 2.75, 3.28],
                        [0.832, 2.117, 0.653]])

        res_keepdims = res.reshape((2, 3, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(
            f(t, aligned_evaluation=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(f(t, aligned_evaluation=False,
                                               keepdims=True).round(3),
                                             res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False).round(3),
            res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=False).round(3),
            res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, aligned_evaluation=False, keepdims=True).round(3),
            res_keepdims)

    def test_evaluation_grid_keepdims_monomial(self):
        """Test behaviour of keepdims with grid evaluation"""

        monomial = Monomial(domain_range=(0, 1), nbasis=3)

        coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

        f = FDataBasis(monomial, coefficients)
        f_keepdims = FDataBasis(monomial, coefficients, keepdims=True)

        np.testing.assert_equal(f.keepdims, False)
        np.testing.assert_equal(f_keepdims.keepdims, True)

        t = np.linspace(0, 1, 4)

        res = np.array([[1., 2., 3.667, 6.],
                        [0.5, 1.111, 2.011, 3.2]])

        res_keepdims = res.reshape((2, 4, 1))

        # Case default behaviour keepdims=False
        np.testing.assert_array_almost_equal(f(t, grid=True).round(3), res)
        np.testing.assert_array_almost_equal(
            f(t, grid=True, keepdims=False).round(3),
            res)

        np.testing.assert_array_almost_equal(
            f(t,  grid=True, keepdims=True).round(3), res_keepdims)

        # Case default behaviour keepdims=True
        np.testing.assert_array_almost_equal(f_keepdims(t, grid=True).round(3),
                                             res_keepdims)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, grid=True, keepdims=False).round(3), res)
        np.testing.assert_array_almost_equal(
            f_keepdims(t, grid=True, keepdims=True).round(3), res_keepdims)

    def test_domain_in_list_monomial(self):
        """Test the evaluation of FDataBasis"""

        for monomial in (Monomial(domain_range=[(0, 1)], nbasis=3),
                         Monomial(domain_range=((0, 1),), nbasis=3),
                         Monomial(domain_range=np.array((0, 1)), nbasis=3),
                         Monomial(domain_range=np.array([(0, 1)]), nbasis=3)):

            coefficients = [[1, 2, 3], [0.5, 1.4, 1.3]]

            f = FDataBasis(monomial, coefficients)

            t = np.linspace(0, 1, 4)

            res = np.array([[1., 2., 3.667, 6.],
                            [0.5, 1.111, 2.011, 3.2]])

            np.testing.assert_array_almost_equal(f(t).round(3), res)
            np.testing.assert_array_almost_equal(f.evaluate(t).round(3), res)


if __name__ == '__main__':
    print()
    unittest.main()
