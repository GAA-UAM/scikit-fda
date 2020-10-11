
from skfda import FDataGrid
from skfda.representation.interpolation import SplineInterpolation
import unittest

import numpy as np


# TODO: Unitest for grids with domain dimension > 1
class TestEvaluationSpline1_1(unittest.TestCase):
    """Test the evaluation of a grid spline interpolation with
    domain and image dimension equal to 1.
    """

    def setUp(self):
        # Data matrix of a datagrid with a dimension of domain and image equal
        # to 1.

        # Matrix of functions (x**2, (9-x)**2)
        self.data_matrix_1_1 = [np.arange(10)**2,
                                np.arange(start=9, stop=-1, step=-1)**2]

    def test_evaluation_linear_simple(self):
        """Test basic usage of evaluation"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10))

        # Test interpolation in nodes
        np.testing.assert_array_almost_equal(
            f(np.arange(10))[..., 0], self.data_matrix_1_1)

        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(
            f([0.5, 1.5, 2.5]),
            np.array([[[0.5],  [2.5],  [6.5]],
                      [[72.5], [56.5], [42.5]]]))

    def test_evaluation_linear_point(self):
        """Test the evaluation of a single point"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10))

        # Test a single point
        np.testing.assert_array_almost_equal(f(5.3).round(1),
                                             np.array([[[28.3]], [[13.9]]]))
        np.testing.assert_array_almost_equal(
            f([3]), np.array([[[9.]], [[36.]]]))
        np.testing.assert_array_almost_equal(
            f((2,)), np.array([[[4.]], [[49.]]]))

    def test_evaluation_linear_grid(self):
        """Test grid evaluation. With domain dimension = 1"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10))

        # Test interpolation in nodes
        np.testing.assert_array_almost_equal(f(np.arange(10))[..., 0],
                                             self.data_matrix_1_1)

        res = np.array([[[0.5],  [2.5],  [6.5]], [[72.5], [56.5], [42.5]]])
        t = [0.5, 1.5, 2.5]

        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(f(t, grid=True), res)
        np.testing.assert_array_almost_equal(f((t,), grid=True), res)
        np.testing.assert_array_almost_equal(f([t], grid=True), res)
        # Single point with grid
        np.testing.assert_array_almost_equal(f(3, grid=True),
                                             np.array([[[9.]], [[36.]]]))

        # Check erroneous axis
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_linear_composed(self):

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10))

        # Evaluate (x**2, (9-x)**2) in (1,8)
        np.testing.assert_array_almost_equal(f([[1], [8]],
                                               aligned=False),
                                             np.array([[[1.]], [[1.]]]))

        t = np.linspace(4, 6, 4)
        np.testing.assert_array_almost_equal(
            f([t, 9 - t], aligned=False).round(2),
            np.array([[[16.], [22.], [28.67], [36.]],
                      [[16.], [22.], [28.67], [36.]]]))

        # Same length than nsample
        t = np.linspace(4, 6, 2)
        np.testing.assert_array_almost_equal(
            f([t, 9 - t], aligned=False).round(2),
            np.array([[[16.], [36.]], [[16.], [36.]]]))

    def test_evaluation_cubic_simple(self):
        """Test basic usage of evaluation"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                      interpolation=SplineInterpolation(3))

        # Test interpolation in nodes
        np.testing.assert_array_almost_equal(f(np.arange(10)).round(1)[..., 0],
                                             self.data_matrix_1_1)

        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(
            f([0.5, 1.5, 2.5]).round(2),
            np.array([[[0.25],  [2.25],  [6.25]],
                      [[72.25], [56.25], [42.25]]]))

    def test_evaluation_cubic_point(self):
        """Test the evaluation of a single point"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                      interpolation=SplineInterpolation(3))

        # Test a single point
        np.testing.assert_array_almost_equal(f(5.3).round(3),
                                             np.array([[[28.09]], [[13.69]]]))

        np.testing.assert_array_almost_equal(
            f([3]).round(3), np.array([[[9.]], [[36.]]]))
        np.testing.assert_array_almost_equal(
            f((2,)).round(3), np.array([[[4.]], [[49.]]]))

    def test_evaluation_cubic_grid(self):
        """Test grid evaluation. With domain dimension = 1"""

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                      interpolation=SplineInterpolation(3))

        t = [0.5, 1.5, 2.5]
        res = np.array([[[0.25],  [2.25],  [6.25]],
                        [[72.25], [56.25], [42.25]]])

        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(f(t, grid=True).round(3), res)
        np.testing.assert_array_almost_equal(f((t,), grid=True).round(3), res)
        np.testing.assert_array_almost_equal(f([t], grid=True).round(3), res)
        # Single point with grid
        np.testing.assert_array_almost_equal(
            f(3, grid=True), np.array([[[9.]], [[36.]]]))

        # Check erroneous axis
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_cubic_composed(self):

        f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                      interpolation=SplineInterpolation(3))

        # Evaluate (x**2, (9-x)**2) in (1,8)
        np.testing.assert_array_almost_equal(
            f([[1], [8]], aligned=False).round(3),
            np.array([[[1.]], [[1.]]]))

        t = np.linspace(4, 6, 4)
        np.testing.assert_array_almost_equal(
            f([t, 9 - t], aligned=False).round(2),
            np.array([[[16.], [21.78], [28.44], [36.]],
                      [[16.], [21.78], [28.44], [36.]]]))

        # Same length than nsample
        t = np.linspace(4, 6, 2)
        np.testing.assert_array_almost_equal(
            f([t, 9 - t], aligned=False).round(3),
            np.array([[[16.], [36.]], [[16.], [36.]]]))

    def test_evaluation_nodes(self):
        """Test interpolation in nodes for all dimensions"""

        for degree in range(1, 6):
            interpolation = SplineInterpolation(degree)

            f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                          interpolation=interpolation)

            # Test interpolation in nodes
            np.testing.assert_array_almost_equal(
                f(np.arange(10)).round(5)[..., 0],
                self.data_matrix_1_1)

    def test_error_degree(self):

        with np.testing.assert_raises(ValueError):
            interpolation = SplineInterpolation(7)
            f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                          interpolation=interpolation)
            f(1)

        with np.testing.assert_raises(ValueError):
            interpolation = SplineInterpolation(0)
            f = FDataGrid(self.data_matrix_1_1, grid_points=np.arange(10),
                          interpolation=interpolation)
            f(1)


class TestEvaluationSpline1_n(unittest.TestCase):
    """Test the evaluation of a grid spline interpolation with
    domain dimension equal to 1 and arbitary image dimension.
    """

    def setUp(self):
        # Data matrix of a datagrid with a dimension of domain and image equal
        # to 1.

        # Matrix of functions (x**2, (9-x)**2)

        self.t = np.arange(10)

        data_1 = np.array([np.arange(10)**2,
                           np.arange(start=9, stop=-1, step=-1)**2])
        data_2 = np.sin(np.pi / 81 * data_1)

        self.data_matrix_1_n = np.dstack((data_1, data_2))

        self.interpolation = SplineInterpolation(interpolation_order=2)

    def test_evaluation_simple(self):
        """Test basic usage of evaluation"""

        f = FDataGrid(self.data_matrix_1_n, grid_points=np.arange(10),
                      interpolation=self.interpolation)

        # Test interpolation in nodes
        np.testing.assert_array_almost_equal(f(self.t), self.data_matrix_1_n)
        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(f([1.5, 2.5, 3.5]),
                                             np.array([[[2.25,  0.087212],
                                                        [6.25,  0.240202],
                                                        [12.25,  0.45773]],
                                                       [[56.25,  0.816142],
                                                        [42.25,  0.997589],
                                                        [30.25,  0.922146]]]
                                                      )
                                             )

    def test_evaluation_point(self):
        """Test the evaluation of a single point"""

        f = FDataGrid(self.data_matrix_1_n, grid_points=np.arange(10),
                      interpolation=self.interpolation)

        # Test a single point
        np.testing.assert_array_almost_equal(f(5.3),
                                             np.array([[[28.09,  0.885526]],
                                                       [[13.69,  0.50697]]]
                                                      )
                                             )

    def test_evaluation_grid(self):
        """Test grid evaluation. With domain dimension = 1"""

        f = FDataGrid(self.data_matrix_1_n, grid_points=np.arange(10),
                      interpolation=SplineInterpolation(2))

        t = [1.5, 2.5, 3.5]
        res = np.array([[[2.25,  0.08721158],
                         [6.25,  0.24020233],
                         [12.25,  0.4577302]],
                        [[56.25,  0.81614206],
                         [42.25,  0.99758925],
                         [30.25,  0.92214607]]])

        # Test evaluation in a list of times
        np.testing.assert_array_almost_equal(f(t, grid=True), res)
        np.testing.assert_array_almost_equal(f((t,), grid=True), res)
        np.testing.assert_array_almost_equal(f([t], grid=True), res)

        # Check erroneous axis
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_composed(self):

        f = FDataGrid(self.data_matrix_1_n, grid_points=self.t,
                      interpolation=self.interpolation)

        # Evaluate (x**2, (9-x)**2) in (1,8)
        np.testing.assert_array_almost_equal(f([[1], [4]],
                                               aligned=False)[0],
                                             f(1)[0])
        np.testing.assert_array_almost_equal(f([[1], [4]],
                                               aligned=False)[1],
                                             f(4)[1])

    def test_evaluation_nodes(self):
        """Test interpolation in nodes for all dimensions"""

        for degree in range(1, 6):
            interpolation = SplineInterpolation(degree)

            f = FDataGrid(self.data_matrix_1_n, grid_points=np.arange(10),
                          interpolation=interpolation)

            # Test interpolation in nodes
            np.testing.assert_array_almost_equal(f(np.arange(10)),
                                                 self.data_matrix_1_n)


class TestEvaluationSplinem_n(unittest.TestCase):
    """Test the evaluation of a grid spline interpolation with
    arbitrary domain dimension and arbitary image dimension.
    """

    def test_evaluation_center_and_extreme_points_linear(self):
        """Test linear interpolation in the middle point of a grid square."""

        dim_codomain = 4
        n_samples = 2

        @np.vectorize
        def coordinate_function(*args):
            _, *domain_indexes, _ = args
            return np.sum(domain_indexes)

        for dim_domain in range(1, 6):
            grid_points = [np.array([0, 1]) for _ in range(dim_domain)]
            data_matrix = np.fromfunction(
                function=coordinate_function,
                shape=(n_samples,) + (2,) * dim_domain + (dim_codomain,))

            f = FDataGrid(data_matrix, grid_points=grid_points)

            evaluation = f([[0.] * dim_domain, [0.5] *
                            dim_domain, [1.] * dim_domain])

            self.assertEqual(evaluation.shape, (n_samples, 3, dim_codomain))

            for i in range(n_samples):
                for j in range(dim_codomain):
                    np.testing.assert_array_almost_equal(
                        evaluation[i, ..., j],
                        [0, dim_domain * 0.5, dim_domain])


if __name__ == '__main__':
    print()
    unittest.main()
