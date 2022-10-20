"""Tests for boxplot functionality."""
import unittest

import numpy as np

from skfda import FDataGrid
from skfda.exploratory.depth import IntegratedDepth
from skfda.exploratory.visualization import Boxplot


class TestBoxplot(unittest.TestCase):
    """Test Boxplot."""

    def test_univariate(self) -> None:
        """Test univariate case."""
        data_matrix = [
            [1, 1, 2, 3, 2.5, 2],
            [0.5, 0.5, 1, 2, 1.5, 1],
            [-1, -1, -0.5, 1, 1, 0.5],  # noqa: WPS204
            [-0.5, -0.5, -0.5, -1, -1, -1],
        ]
        grid_points = [0, 2, 4, 6, 8, 10]
        fd = FDataGrid(data_matrix, grid_points)
        fdataBoxplot = Boxplot(fd, depth_method=IntegratedDepth())
        np.testing.assert_array_equal(
            fdataBoxplot.median.ravel(),
            np.array([-1, -1, -0.5, 1, 1, 0.5]),
        )
        np.testing.assert_array_equal(
            fdataBoxplot.central_envelope[0].ravel(),
            np.array([-1, -1, -0.5, -1, -1, -1]),
        )
        np.testing.assert_array_equal(
            fdataBoxplot.central_envelope[1].ravel(),
            np.array([-0.5, -0.5, -0.5, 1, 1, 0.5]),
        )
        np.testing.assert_array_equal(
            fdataBoxplot.non_outlying_envelope[0].ravel(),
            np.array([-1, -1, -0.5, -1, -1, -1]),
        )
        np.testing.assert_array_equal(
            fdataBoxplot.non_outlying_envelope[1].ravel(),
            np.array([-0.5, -0.5, -0.5, 1, 1, 0.5]),
        )
        self.assertEqual(len(fdataBoxplot.envelopes), 1)
        np.testing.assert_array_equal(
            fdataBoxplot.envelopes[0],
            fdataBoxplot.central_envelope,
        )
        np.testing.assert_array_equal(
            fdataBoxplot.outliers,
            np.array([True, True, False, False]),
        )


if __name__ == '__main__':
    unittest.main()
