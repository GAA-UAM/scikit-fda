"""Tests for depth measures."""
import unittest

import numpy as np

import skfda
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth


class TestsDepthSameCurves(unittest.TestCase):
    """Test behavior when all curves in the distribution are the same."""

    def setUp(self) -> None:
        """Define the dataset."""
        data_matrix = np.tile([1, 2, 3, 4], (5, 1))

        self.fd = skfda.FDataGrid(data_matrix)

    def test_integrated_equal(self) -> None:
        """
        Test the Fraiman-Muñiz depth.

        Results should be equal to the minimum depth (0.5).

        """
        depth = IntegratedDepth()

        np.testing.assert_almost_equal(
            depth(self.fd),
            [0.5, 0.5, 0.5, 0.5, 0.5],
        )

    def test_modified_band_depth_equal(self) -> None:
        """
        Test MBD.

        Results should be equal to the maximum depth (1).

        """
        depth = ModifiedBandDepth()

        np.testing.assert_almost_equal(
            depth(self.fd),
            [1, 1, 1, 1, 1],
        )
