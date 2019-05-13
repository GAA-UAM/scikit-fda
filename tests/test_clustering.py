import unittest
import numpy as np

from skfda.grid import FDataGrid
from skfda.clustering import KMeans, FuzzyKMeans


class TestClustering(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        init = np.array([[0, 0, 0, 0, 0, 0], [2, 1, -1, 0.5, 0, -0.5]])
        init_fd = FDataGrid(init, sample_points)
        kmeans = KMeans(init=init_fd)
        kmeans.fit(fd)
        np.testing.assert_array_equal(kmeans.predict(fd),
                                      np.array([[0], [0], [0], [1]]))
        np.testing.assert_allclose(kmeans.transform(fd),
                                   np.array([[[2.98142397],
                                              [9.23534876]],
                                             [[0.68718427],
                                              [6.50960828]],
                                             [[3.31243449],
                                              [4.39222798]],
                                             [[6.49679408],
                                              [0.]]]))
        centers = FDataGrid(data_matrix=np.array(
            [[0.16666667, 0.16666667, 0.83333333, 2., 1.66666667, 1.16666667],
             [-0.5, -0.5, -0.5, -1., -1., -1.]]),
            sample_points=sample_points)
        np.testing.assert_array_almost_equal(
            kmeans.cluster_centers_.data_matrix,
            centers.data_matrix)
        np.testing.assert_array_equal(kmeans.n_iter_, np.array([3.]))

    def test_kmeans_multivariate(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        kmeans = KMeans()
        kmeans.fit(fd)
        np.testing.assert_array_equal(kmeans.predict(fd),
                                      np.array([[1, 0],
                                                [1, 1],
                                                [0, 0]]))
        np.testing.assert_allclose(kmeans.transform(fd),
                                   np.array([[[4.89897949, 0.12247449],
                                              [1.22474487, 0.46368092]],
                                             [[2.44948974, 0.5845226],
                                              [1.22474487, 0.]],
                                             [[0., 0.12247449],
                                              [3.67423461, 0.70592729]]]))
        centers = FDataGrid(data_matrix=np.array(
            [[[3, 0.25], [4, 0.35], [5, 0.45], [6, 0.55]],
             [[1.5, 0.5], [2.5, 0.6], [3.5, 0.7], [4.5, 0.7]]]),
            sample_points=sample_points)
        np.testing.assert_allclose(kmeans.cluster_centers_.data_matrix,
                                   centers.data_matrix)
        np.testing.assert_array_equal(kmeans.n_iter_, np.array([2., 2.]))

    def test_fuzzy_kmeans_univariate(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        sample_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, sample_points)
        fuzzy_kmeans = FuzzyKMeans()
        fuzzy_kmeans.fit(fd)
        np.testing.assert_array_equal(fuzzy_kmeans.predict(fd),
                                      np.array([[[0.965, 0.035]],
                                                [[0.94, 0.06]],
                                                [[0.227, 0.773]],
                                                [[0.049, 0.951]]]))
        np.testing.assert_allclose(fuzzy_kmeans.transform(fd),
                                   np.array([[[2.22634867],
                                              [62.07348409]],
                                             [[1.67440984],
                                              [26.28261015]],
                                             [[23.57706928],
                                              [6.93094191]],
                                             [[60.44654168],
                                              [3.09632834]]]))
        centers = FDataGrid(data_matrix=np.array(
            [[0.7065429, 0.7065429, 1.45512711, 2.46702433,
              1.98146141, 1.48209637],
             [-0.69458976, -0.69458976, -0.49444057, -0.19702289,
              -0.19861085, -0.39836226]]), sample_points=sample_points)
        np.testing.assert_allclose(fuzzy_kmeans.cluster_centers_.data_matrix,
                                   centers.data_matrix)
        np.testing.assert_array_equal(fuzzy_kmeans.n_iter_, np.array([67.]))

    def test_fuzzy_kmeans_multivariate(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        init = np.array([[[3, 0], [5, 0], [2, 0], [4, 0]],
                         [[0, 0], [0, 1], [0, 0], [0, 1]]])
        init_fd = FDataGrid(init, sample_points)
        fuzzy_kmeans = FuzzyKMeans(init=init_fd)
        fuzzy_kmeans.fit(fd)
        np.testing.assert_array_equal(fuzzy_kmeans.predict(fd),
                                      np.array([[[0., 1.],
                                                 [0.5, 0.5]],
                                                [[1., 0.],
                                                 [0.5, 0.5]],
                                                [[0.8, 0.2],
                                                 [0.5, 0.5]]]))
        np.testing.assert_allclose(fuzzy_kmeans.transform(fd),
                                   np.array([[[25., 1.26333333],
                                              [126.33333333, 1.26333333]],
                                             [[25., 2.45833333],
                                              [126.33333333, 2.45833333]],
                                             [[6., 0.78333333],
                                              [24., 0.78333333]]]))
        centers = FDataGrid(data_matrix=np.array(
            [[[2, 0], [3, 0], [4, 0], [5, 0]],
             [[1, 0], [2, 0], [3, 0], [4, 0]]]), sample_points=sample_points)
        np.testing.assert_allclose(fuzzy_kmeans.cluster_centers_.data_matrix,
                                   centers.data_matrix)
        np.testing.assert_array_equal(fuzzy_kmeans.n_iter_,
                                      np.array([2., 2.]))


if __name__ == '__main__':
    print()
    unittest.main()
