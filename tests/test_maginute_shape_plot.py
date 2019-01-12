import unittest
import numpy as np

from fda.grid import FDataGrid
from fda.magnitude_shape_plot import *

class TestBoxplot(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

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

    # def test_magnitude_shape_plot(self):
    #     ???


if __name__ == '__main__':
    print()
    unittest.main()
