import unittest

import numpy as np
from skfda import FDataGrid
from skfda.exploratory.depth import modified_band_depth
from skfda.exploratory.outliers import directional_outlyingness_stats


class TestsDirectionalOutlyingness(unittest.TestCase):

    def test_directional_outlyingness(self):
        data_matrix = [[[0.3], [0.4], [0.5], [0.6]],
                       [[0.5], [0.6], [0.7], [0.7]],
                       [[0.2], [0.3], [0.4], [0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        stats = directional_outlyingness_stats(
            fd, depth_method=modified_band_depth)
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


if __name__ == '__main__':
    print()
    unittest.main()
