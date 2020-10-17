import skfda
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth
import unittest
import numpy as np


class TestsDepthSameCurves(unittest.TestCase):

    def setUp(self):
        data_matrix = [[1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4]]

        self.fd = skfda.FDataGrid(data_matrix)

    def test_integrated_equal(self):

        depth = IntegratedDepth()

        np.testing.assert_almost_equal(
            depth(self.fd), [0.5, 0.5, 0.5, 0.5, 0.5])

    def test_modified_band_depth_equal(self):

        depth = ModifiedBandDepth()

        np.testing.assert_almost_equal(
            depth(self.fd), [1, 1, 1, 1, 1])
