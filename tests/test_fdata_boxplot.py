import unittest
import numpy as np

from skfda import FDataGrid
from skfda.exploratory.depth import band_depth, fraiman_muniz_depth
from skfda.exploratory.visualization.boxplot import Boxplot, SurfaceBoxplot
import matplotlib.pyplot as plt


class TestBoxplot(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_fdboxplot_multivariate(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        fdataBoxplot = Boxplot(fd, prob=[0.75, 0.5, 0.25])
        np.testing.assert_array_equal(fdataBoxplot.median,
                                      np.array([[2., 3., 4., 5.],
                                                [0.3, 0.4, 0.5, 0.6]]))
        np.testing.assert_array_equal(fdataBoxplot.central_envelope, np.array(
            [[[2., 3., 4., 5.], [1., 2., 3., 4.]],
             [[0.5, 0.6, 0.7, 0.7],
              [0.3, 0.4, 0.5, 0.6]]]))
        np.testing.assert_array_equal(fdataBoxplot.outlying_envelope, np.array(
            [[[3., 4., 5., 6.], [1., 2., 3., 4.]],
             [[0.5, 0.6, 0.7, 0.7],
              [0.2, 0.3, 0.4, 0.5]]]))
        np.testing.assert_array_equal(fdataBoxplot.central_regions, np.array(
            [[[3., 4., 5., 6.], [1., 2., 3., 4.]],
             [[2., 3., 4., 5.], [1., 2., 3., 4.]],
             [[2., 3., 4., 5.], [2., 3., 4., 5.]],
             [[0.5, 0.6, 0.7, 0.7],
              [0.2, 0.3, 0.4, 0.5]],
             [[0.5, 0.6, 0.7, 0.7],
              [0.3, 0.4, 0.5, 0.6]],
             [[0.3, 0.4, 0.5, 0.6],
              [0.3, 0.4, 0.5, 0.6]]]))
        np.testing.assert_array_equal(fdataBoxplot.outliers,
                                      np.array([[0., 0., 0.],
                                                [0., 0., 0.]]))

    def test_fdboxplot_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        fdataBoxplot = Boxplot(fd, method=fraiman_muniz_depth)
        np.testing.assert_array_equal(fdataBoxplot.median, np.array(
            [[-1., -1., -0.5, 1., 1., 0.5]]))
        np.testing.assert_array_equal(fdataBoxplot.central_envelope, np.array(
            [[[-0.5, -0.5, -0.5, 1., 1., 0.5],
              [-1., -1., -0.5, -1., -1., -1.]]]))
        np.testing.assert_array_equal(fdataBoxplot.outlying_envelope, np.array(
            [[[0.25, 0.25, -0.5, 3., 2.5, 2.],
              [-1., -1., -0.5, -1., -1., -1.]]]))
        np.testing.assert_array_equal(fdataBoxplot.central_regions, np.array(
            [[[-0.5, -0.5, -0.5, 1., 1., 0.5],
              [-1., -1., -0.5, -1., -1., -1.]]]))
        np.testing.assert_array_equal(fdataBoxplot.outliers,
                                      np.array([[1., 1., 0., 0.]]))

    def test_surface_boxplot(self):
        data_matrix = [
            [[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
            [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]],
            [[[5, 8], [5, 2], [1, 6]], [[5, 20], [0.3, 5], [7, 1]]]]
        sample_points = [[2, 4], [3, 6, 8]]
        fd = FDataGrid(data_matrix, sample_points)
        plt.figure()
        fdataBoxplot = SurfaceBoxplot(fd, method=band_depth)
        np.testing.assert_array_equal(fdataBoxplot.median,
                                      np.array([[[1., 0.3, 1.], [2., 0.4, 2.]],
                                                [[4., 1.5, 3.],
                                                 [8., 2., 9.]]]))
        np.testing.assert_array_equal(fdataBoxplot.central_envelope, np.array(
            [[[[2., 0.5, 2.], [3., 0.6, 3.]],
              [[1., 0.3, 1.], [2., 0.4, 2.]]],
             [[[10., 3., 10.], [12., 3., 15.]],
              [[4., 1.5, 3.], [8., 2., 9.]]]]))
        np.testing.assert_array_equal(fdataBoxplot.outlying_envelope, np.array(
            [[[[3.5, 0.8, 2.], [4.5, 0.6, 4.5]],
              [[1., 0.3, 1.], [2., 0.3, 2.]]],
             [[[10., 3., 10.], [18., 4.5, 15.]],
              [[4., 1.5, 3.], [8., 2., 1.]]]]))


if __name__ == '__main__':
    print()
    unittest.main()
