import unittest
import numpy as np

from fda.grid import FDataGrid
from fda.depth_measures import band_depth, modified_band_depth, Fraiman_Muniz_depth, directional_outlyingness


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

    def test_Fraiman_Muniz_band_depth_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        sample_depth, pointwise_sample_depth = Fraiman_Muniz_depth(fd, pointwise=True)
        np.testing.assert_allclose(sample_depth,
                                   np.array([[0.5], [0.75], [0.91666667], [0.875]]))
        np.testing.assert_array_equal(pointwise_sample_depth,
                                      np.array([[[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                                                [[0.75], [0.75], [0.75], [0.75], [0.75], [0.75]],
                                                [[0.75], [0.75], [1.], [1.], [1.], [1.]],
                                                [[1.], [1.], [1.], [0.75], [0.75], [0.75]]]))

    def test_Fraiman_Muniz_depth_multivariate(self):
        data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
                       [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        sample_points = [[2, 4], [3, 6, 8]]
        fd = FDataGrid(data_matrix, sample_points)
        sample_depth, pointwise_sample_depth = Fraiman_Muniz_depth(fd, pointwise=True)
        np.testing.assert_array_equal(sample_depth, np.array([[1., 1.], [0.5, 0.5]]))
        np.testing.assert_allclose(pointwise_sample_depth,
                                   np.array([[[[1., 1.], [1., 1.], [1., 1.]],
                                              [[1., 1.], [1., 1.], [1., 1.]]],
                                             [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                                              [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]]))

    def test_directional_outlyingness(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        mean_dir_outl, variation_dir_outl = directional_outlyingness(fd)
        np.testing.assert_allclose(mean_dir_outl,
                                   np.array([[0., 0.], [0.19683896, 0.03439261], [0.49937617, -0.02496881]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(variation_dir_outl,
                                   np.array([0.00000000e+00, 7.15721232e-05, 4.81482486e-35]))


if __name__ == '__main__':
    print()
    unittest.main()
