import unittest
import numpy as np

from skfda import FDataGrid
from skfda.exploratory.depth import band_depth, modified_band_depth, fraiman_muniz_depth


class TestDepthMeasures(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_band_depth_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        depth = band_depth(fd)
        np.testing.assert_allclose(depth, np.array([[0.5], [0.83333333], [0.5], [0.5]]))

    def test_band_depth_multivariate(self):
        data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
                       [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        sample_points = [[2, 4], [3, 6, 8]]
        fd = FDataGrid(data_matrix, sample_points)
        depth = band_depth(fd)
        np.testing.assert_array_equal(depth, np.array([[1., 1.], [1., 1.]]))

    def test_modified_band_depth_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        depth = modified_band_depth(fd, pointwise=True)
        sample_depth = depth[0]
        pointwise_sample_depth = depth[1]
        np.testing.assert_allclose(sample_depth,
                                   np.array([[0.5], [0.83333333], [0.72222222], [0.66666667]]))
        np.testing.assert_allclose(pointwise_sample_depth,
                                   np.array([[[0.5], [0.5], [0.5],
                                              [0.5], [0.5], [0.5]],
                                             [[0.83333333], [0.83333333], [0.83333333],
                                              [0.83333333], [0.83333333], [0.83333333]],
                                             [[0.5], [0.5], [0.83333333],
                                              [0.83333333], [0.83333333], [0.83333333]],
                                             [[0.83333333], [0.83333333], [0.83333333],
                                              [0.5], [0.5], [0.5]]]))

    def test_modified_band_depth_multivariate(self):
        data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
                       [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        sample_points = [[2, 4], [3, 6, 8]]
        fd = FDataGrid(data_matrix, sample_points)
        depth = modified_band_depth(fd, pointwise=True)
        sample_depth = depth[0]
        pointwise_sample_depth = depth[1]
        np.testing.assert_array_equal(sample_depth, np.array([[1., 1.], [1., 1.]]))
        np.testing.assert_array_equal(pointwise_sample_depth,
                                      np.array([[[[1., 1.], [1., 1.], [1., 1.]],
                                                 [[1., 1.], [1., 1.], [1., 1.]]],
                                                [[[1., 1.], [1., 1.], [1., 1.]],
                                                 [[1., 1.], [1., 1.], [1., 1.]]]]))

    def test_fraiman_muniz_band_depth_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        sample_depth, pointwise_sample_depth = fraiman_muniz_depth(fd, pointwise=True)
        np.testing.assert_allclose(sample_depth,
                                   np.array([[0.5], [0.75], [0.91666667], [0.875]]))
        np.testing.assert_array_equal(pointwise_sample_depth,
                                      np.array([[[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                                                [[0.75], [0.75], [0.75], [0.75], [0.75], [0.75]],
                                                [[0.75], [0.75], [1.], [1.], [1.], [1.]],
                                                [[1.], [1.], [1.], [0.75], [0.75], [0.75]]]))

    def test_fraiman_muniz_depth_multivariate(self):
        data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
                       [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        sample_points = [[2, 4], [3, 6, 8]]
        fd = FDataGrid(data_matrix, sample_points)
        sample_depth, pointwise_sample_depth = fraiman_muniz_depth(fd, pointwise=True)
        np.testing.assert_array_equal(sample_depth, np.array([[1., 1.], [0.5, 0.5]]))
        np.testing.assert_allclose(pointwise_sample_depth,
                                   np.array([[[[1., 1.], [1., 1.], [1., 1.]],
                                              [[1., 1.], [1., 1.], [1., 1.]]],
                                             [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                                              [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]]))


if __name__ == '__main__':
    print()
    unittest.main()
