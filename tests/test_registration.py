import unittest

import numpy as np

import matplotlib.pyplot as plt

from fda.grid import FDataGrid, GridSplineInterpolator
from fda.datasets import make_multimodal_samples
from fda.metrics import metric
from fda.registration import normalize_warping, invert_warping


class TestWarping(unittest.TestCase):
    """Test warpings functions"""


    def setUp(self):
        """Initialization of samples"""

        self.time = np.linspace(-1, 1, 50)
        interpolator = GridSplineInterpolator(3, monotone=True)
        self.polynomial = FDataGrid([self.time**3, self.time**5],
                                    self.time, interpolator=interpolator)

    def test_invert_warping(self):

        inverse = invert_warping(self.polynomial)

        # Check if identity
        id = self.polynomial.compose(inverse)

        np.testing.assert_array_almost_equal([self.time, self.time],
                                             id.data_matrix[..., 0],
                                             decimal=3)

    def test_standart_normalize_warpig(self):
        """Test normalization to (0, 1)"""

        normalized = normalize_warping(self.polynomial)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(0, 1)])

        np.testing.assert_array_almost_equal(normalized.sample_points[0],
                                             np.linspace(0, 1, 50))

        np.testing.assert_array_equal(normalized(0), [[0], [0]])

        np.testing.assert_array_equal(normalized(1), [[1], [1]])

    def test_normalize_warpig(self):
        """Test normalization to (a, b)"""
        a = -4
        b = 3
        normalized = normalize_warping(self.polynomial, a=a, b=b)

        # Test new domain range (0, 1)
        np.testing.assert_array_equal(normalized.domain_range, [(a, b)])

        np.testing.assert_array_almost_equal(normalized.sample_points[0],
                                             np.linspace(a, b, 50))

        np.testing.assert_array_equal(normalized(a), [[a], [a]])

        np.testing.assert_array_equal(normalized(b), [[b], [b]])







if __name__ == '__main__':
    print()
    unittest.main()
