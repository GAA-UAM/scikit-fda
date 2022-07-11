import unittest

import numpy as np

from skfda import FDataGrid
from skfda.exploratory.depth.multivariate import SimplicialDepth
from skfda.exploratory.outliers import (
    MSPlotOutlierDetector,
    directional_outlyingness_stats,
)


class TestsDirectionalOutlyingness(unittest.TestCase):

    def test_directional_outlyingness(self):
        data_matrix = [[[0.3], [0.4], [0.5], [0.6]],
                       [[0.5], [0.6], [0.7], [0.7]],
                       [[0.2], [0.3], [0.4], [0.5]]]
        grid_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, grid_points)
        stats = directional_outlyingness_stats(
            fd, multivariate_depth=SimplicialDepth())
        np.testing.assert_allclose(stats.directional_outlyingness,
                                   np.array([[[0.],
                                              [0.],
                                              [0.],
                                              [0.]],

                                             [[0.5],
                                              [0.5],
                                              [0.5],
                                              [0.5]],

                                             [[-0.5],
                                              [-0.5],
                                              [-0.5],
                                              [-0.5]]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(stats.mean_directional_outlyingness,
                                   np.array([[0.],
                                             [0.5],
                                             [-0.5]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(stats.variation_directional_outlyingness,
                                   np.array([0., 0., 0.]), atol=1e-6)

    def test_asymptotic_formula(self):
        data_matrix = [[1, 1, 2, 3, 2.5, 2],
                       [0.5, 0.5, 1, 2, 1.5, 1],
                       [-1, -1, -0.5, 1, 1, 0.5],
                       [-0.5, -0.5, -0.5, -1, -1, -1]]
        grid_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, grid_points)
        out_detector = MSPlotOutlierDetector(
            _force_asymptotic=True)
        prediction = out_detector.fit_predict(fd)
        np.testing.assert_allclose(prediction,
                                   np.array([1, 1, 1, 1]))


if __name__ == '__main__':
    print()
    unittest.main()
