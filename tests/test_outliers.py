import unittest

import numpy as np
from skfda import FDataGrid
from skfda.exploratory.outliers import directional_outlyingness_stats


class TestsDirectionalOutlyingness(unittest.TestCase):

    def test_directional_outlyingness(self):
        data_matrix = [[[0.3], [0.4], [0.5], [0.6]],
                       [[0.5], [0.6], [0.7], [0.7]],
                       [[0.2], [0.3], [0.4], [0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        stats = directional_outlyingness_stats(fd)
        np.testing.assert_allclose(stats.directional_outlyingness,
                                   np.array([[[0.],
                                              [0.],
                                              [0.],
                                              [0.]],

                                             [[1.],
                                              [1.],
                                              [1.],
                                              [1.]],

                                             [[-0.2],
                                              [-0.2],
                                              [-0.2],
                                              [-0.2]]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(stats.mean_directional_outlyingness,
                                   np.array([[0.],
                                             [1.5],
                                             [-0.3]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(stats.variation_directional_outlyingness,
                                   np.array([0., 0.375, 0.015]))


if __name__ == '__main__':
    print()
    unittest.main()
