"""Tests for clustering methods."""
import unittest

import numpy as np

from skfda.ml.clustering import FuzzyCMeans, KMeans
from skfda.representation.grid import FDataGrid


class TestKMeans(unittest.TestCase):
    """Test the KMeans clustering method."""

    def test_univariate(self) -> None:
        """Test with univariate functional data."""
        data_matrix = [
            [1, 1, 2, 3, 2.5, 2],
            [0.5, 0.5, 1, 2, 1.5, 1],
            [-1, -1, -0.5, 1, 1, 0.5],  # noqa: WPS204
            [-0.5, -0.5, -0.5, -1, -1, -1],
        ]
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
                [6.49679408, 0],
            ]),
        )
        np.testing.assert_array_equal(
            kmeans.predict(fd),
            np.array([0, 0, 0, 1]),
        )
        np.testing.assert_allclose(
            kmeans.transform(fd),
            np.array([
                [2.98142397, 9.23534876],
                [0.68718427, 6.50960828],
                [3.31243449, 4.39222798],
                [6.49679408, 0],
            ]),
        )
        centers = FDataGrid(
            data_matrix=np.array([
                [  # noqa: WPS317
                    0.16666667, 0.16666667, 0.83333333,
                    2.0, 1.66666667, 1.16666667,
                ],
                [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0],
            ]),
            grid_points=grid_points,
        )
        np.testing.assert_array_almost_equal(
            kmeans.cluster_centers_.data_matrix,
            centers.data_matrix,
        )
        np.testing.assert_allclose(kmeans.score(fd), np.array([-20.33333333]))
        np.testing.assert_array_equal(kmeans.n_iter_, np.array([3.0]))


class TestFuzzyCMeans(unittest.TestCase):
    """Test the FuzzyCMeans clustering method."""

    def test_univariate(self) -> None:
        """Test with univariate functional data."""
        data_matrix = [
            [1, 1, 2, 3, 2.5, 2],
            [0.5, 0.5, 1, 2, 1.5, 1],
            [-1, -1, -0.5, 1, 1, 0.5],  # noqa: WPS204
            [-0.5, -0.5, -0.5, -1, -1, -1],
        ]
        grid_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, grid_points)
        fuzzy_kmeans = FuzzyCMeans[FDataGrid]()
        fuzzy_kmeans.fit(fd)
        np.testing.assert_array_equal(
            fuzzy_kmeans.predict_proba(fd).round(3),
            np.array([
                [0.965, 0.035],
                [0.94, 0.06],
                [0.227, 0.773],
                [0.049, 0.951],
            ]),
        )
        np.testing.assert_allclose(
            fuzzy_kmeans.transform(fd).round(3),
            np.array([
                [1.492, 7.879],
                [1.294, 5.127],
                [4.856, 2.633],
                [7.775, 1.759],
            ]),
        )
        centers = np.array([
            [0.707, 0.707, 1.455, 2.467, 1.981, 1.482],
            [-0.695, -0.695, -0.494, -0.197, -0.199, -0.398],
        ])
        np.testing.assert_allclose(
            fuzzy_kmeans.cluster_centers_.data_matrix[..., 0],
            centers,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            fuzzy_kmeans.score(fd),
            np.array([-12.025179]),
        )
        self.assertEqual(fuzzy_kmeans.n_iter_, 19)


if __name__ == '__main__':
    unittest.main()
