from skfda.ml.clustering import KMeans, FuzzyCMeans
from skfda.representation.grid import FDataGrid
import unittest

import numpy as np


class TestClustering(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        grid_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, grid_points)
        init = np.array([[0, 0, 0, 0, 0, 0], [2, 1, -1, 0.5, 0, -0.5]])
        init_fd = FDataGrid(init, grid_points)
        kmeans = KMeans(init=init_fd)
        distances_to_centers = kmeans.fit_transform(fd)
        np.testing.assert_allclose(
            distances_to_centers,
            np.array([
                [2.98142397, 9.23534876],
                [0.68718427, 6.50960828],
                [3.31243449, 4.39222798],
                [6.49679408, 0.0],
            ]),
        )
        np.testing.assert_array_equal(kmeans.predict(fd),
                                      np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(kmeans.transform(fd),
                                   np.array([[2.98142397, 9.23534876],
                                             [0.68718427, 6.50960828],
                                             [3.31243449, 4.39222798],
                                             [6.49679408, 0.]]))
        centers = FDataGrid(data_matrix=np.array(
            [[0.16666667, 0.16666667, 0.83333333, 2., 1.66666667, 1.16666667],
             [-0.5, -0.5, -0.5, -1., -1., -1.]]),
            grid_points=grid_points)
        np.testing.assert_array_almost_equal(
            kmeans.cluster_centers_.data_matrix,
            centers.data_matrix)
        np.testing.assert_allclose(kmeans.score(fd), np.array([-20.33333333]))
        np.testing.assert_array_equal(kmeans.n_iter_, np.array([3.]))

    # def test_kmeans_multivariate(self):
    #     data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
    #                    [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
    #                    [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
    #     grid_points = [2, 4, 6, 8]
    #     fd = FDataGrid(data_matrix, grid_points)
    #     kmeans = KMeans()
    #     kmeans.fit(fd)
    #     np.testing.assert_array_equal(kmeans.predict(fd),
    #                                   np.array([[1, 1],
    #                                             [1, 1],
    #                                             [0, 0]]))
    #     np.testing.assert_allclose(kmeans.transform(fd),
    #                                np.array([[[4.89897949, 0.24494897],
    #                                           [1.22474487, 0.23184046]],
    #                                          [[2.44948974, 0.70592729],
    #                                           [1.22474487, 0.23184046]],
    #                                          [[0., 0.],
    #                                           [3.67423461, 0.47478065]]]))
    #     centers = FDataGrid(data_matrix=np.array(
    #         [[[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]],
    #          [[1.5, 0.4], [2.5, 0.5], [3.5, 0.6], [4.5, 0.65]]]),
    #         grid_points=grid_points)
    #     np.testing.assert_allclose(kmeans.cluster_centers_.data_matrix,
    #                                centers.data_matrix)
    #     np.testing.assert_allclose(kmeans.score(fd), np.array([-3., -0.1075]))
    #     np.testing.assert_array_equal(kmeans.n_iter_, np.array([2., 2.]))

    def test_fuzzy_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        grid_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, grid_points)
        fuzzy_kmeans = FuzzyCMeans()
        fuzzy_kmeans.fit(fd)
        np.testing.assert_array_equal(fuzzy_kmeans.predict(fd).round(3),
                                      np.array([[0.965, 0.035],
                                                [0.94, 0.06],
                                                [0.227, 0.773],
                                                [0.049, 0.951]]))
        np.testing.assert_allclose(fuzzy_kmeans.transform(fd).round(3),
                                   np.array([[1.492, 7.879],
                                             [1.294, 5.127],
                                             [4.856, 2.633],
                                             [7.775, 1.759]]))
        centers = np.array([[0.707, 0.707, 1.455, 2.467, 1.981, 1.482],
                            [-0.695, -0.695, -0.494, -0.197, -0.199, -0.398]])
        np.testing.assert_allclose(
            fuzzy_kmeans.cluster_centers_.data_matrix[..., 0].round(3),
            centers)
        np.testing.assert_allclose(fuzzy_kmeans.score(fd),
                                   np.array([-12.025179]))
        self.assertEqual(fuzzy_kmeans.n_iter_, 19)

    # def test_fuzzy_kmeans_multivariate(self):
    #     data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
    #                    [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
    #                    [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
    #     grid_points = [2, 4, 6, 8]
    #     fd = FDataGrid(data_matrix, grid_points)
    #     init = np.array([[[3, 0], [5, 0], [2, 0], [4, 0]],
    #                      [[0, 0], [0, 1], [0, 0], [0, 1]]])
    #     init_fd = FDataGrid(init, grid_points)
    #     fuzzy_kmeans = FuzzyKMeans(init=init_fd)
    #     fuzzy_kmeans.fit(fd)
    #     np.testing.assert_array_equal(fuzzy_kmeans.predict(fd),
    #                                   np.array([[[0., 1.],
    #                                              [0.5, 0.5]],
    #                                             [[1., 0.],
    #                                              [0.5, 0.5]],
    #                                             [[0.8, 0.2],
    #                                              [0.5, 0.5]]]))
    #     np.testing.assert_allclose(fuzzy_kmeans.transform(fd),
    #                                np.array([[[25., 1.26333333],
    #                                           [126.33333333, 1.26333333]],
    #                                          [[25., 2.45833333],
    #                                           [126.33333333, 2.45833333]],
    #                                          [[6., 0.78333333],
    #                                           [24., 0.78333333]]]))
    #     centers = FDataGrid(data_matrix=np.array(
    #         [[[2, 0], [3, 0], [4, 0], [5, 0]],
    #          [[1, 0], [2, 0], [3, 0], [4, 0]]]), grid_points=grid_points)
    #     np.testing.assert_allclose(fuzzy_kmeans.cluster_centers_.data_matrix,
    #                                centers.data_matrix)
    #     np.testing.assert_allclose(fuzzy_kmeans.score(fd), np.array(
    #         [-1.66211111e+04, -8.25302500e+00]))
    #     np.testing.assert_array_equal(fuzzy_kmeans.n_iter_,
    #                                   np.array([2., 2.]))


if __name__ == '__main__':
    print()
    unittest.main()
