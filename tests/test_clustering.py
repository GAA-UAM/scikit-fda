import unittest
import numpy as np

from fda.grid import FDataGrid
from fda.clustering import KMeans, FuzzyKMeans


class TestClustering(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        kmeans = KMeans()
        kmeans.fit(fd, init=np.array([[[0, 0, 0, 0, 0, 0],
                                       [2, 1, -1, 0.5, 0, -0.5]]]))
        np.testing.assert_array_equal(kmeans.clustering_values,
                                      np.array([[1], [1], [1], [0]]))
        np.testing.assert_array_almost_equal(kmeans.centers, np.array(
            [[[-0.5, -0.5, -0.5, -1., -1., -1.],
              [0.16666667, 0.16666667, 0.83333333, 2., 1.66666667,
               1.16666667]]]))
        np.testing.assert_array_equal(kmeans.n_iter, np.array([2.]))

    def test_kmeans_multivariate(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        kmeans = KMeans()
        kmeans.fit(fd)
        np.testing.assert_array_equal(kmeans.clustering_values,
                                      np.array([[0, 1],
                                                [0, 0],
                                                [1, 1]]))
        np.testing.assert_allclose(kmeans.centers, np.array(
            [[[1.5, 2.5, 3.5, 4.5],
              [3., 4., 5., 6.]],
             [[0.5, 0.6, 0.7, 0.7],
              [0.25, 0.35, 0.45, 0.55]]]))
        np.testing.assert_array_equal(kmeans.n_iter, np.array([2., 2.]))

    def test_fuzzy_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        fuzzy_kmeans = FuzzyKMeans()
        fuzzy_kmeans.fit(fd)
        np.testing.assert_array_equal(fuzzy_kmeans.membership_values,
                                      np.array([[[0.035, 0.965]],
                                                [[0.06, 0.94]],
                                                [[0.773, 0.227]],
                                                [[0.951, 0.049]]]))
        np.testing.assert_allclose(fuzzy_kmeans.centers, np.array(
            [[[-0.69458976, -0.69458976, -0.49444057, -0.19702289,
               -0.19861085, -0.39836226],
              [0.7065429, 0.7065429, 1.45512711, 2.46702433,
               1.98146141, 1.48209637]]]))
        np.testing.assert_array_equal(fuzzy_kmeans.n_iter, np.array([66.]))

    def test_fuzzy_kmeans_multivariate(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        fuzzy_kmeans = FuzzyKMeans()
        fuzzy_kmeans.fit(fd, init=np.array([[[0, 0, 0, 0], [0, 0, 0, 0]],
                                            [[3, 5, 2, 4], [0, 1, 0, 1]]]))
        np.testing.assert_array_equal(fuzzy_kmeans.membership_values,
                                      np.array([[[0.987, 0.013],
                                                 [0.072, 0.928]],
                                                [[0.5, 0.5],
                                                 [1., 0.]],
                                                [[0.013, 0.987],
                                                 [0.027, 0.973]]]))
        np.testing.assert_allclose(fuzzy_kmeans.centers, np.array(
            [[[1.20439134, 2.20439134, 3.20439134, 4.20439134],
              [2.79560869, 3.79560869, 4.79560869, 5.79560869]],
             [[0.49875695, 0.59875695, 0.69875695, 0.69934278],
              [0.24762741, 0.34762741, 0.44762741, 0.54762741]]]))

        np.testing.assert_array_equal(fuzzy_kmeans.n_iter,
                                      np.array([100., 23.]))


if __name__ == '__main__':
    print()
    unittest.main()
