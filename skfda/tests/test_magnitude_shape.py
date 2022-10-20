"""Tests for Magnitude-Shape plot."""
import unittest

import numpy as np

from skfda.datasets import fetch_weather
from skfda.exploratory.depth.multivariate import SimplicialDepth
from skfda.exploratory.visualization import MagnitudeShapePlot


class TestMagnitudeShapePlot(unittest.TestCase):
    """Test MS plot."""

    def test_magnitude_shape_plot(self) -> None:
        """Test MS plot properties."""
        fd = fetch_weather()["data"]
        fd_temperatures = fd.coordinates[0]
        msplot = MagnitudeShapePlot(
            fd_temperatures,
            multivariate_depth=SimplicialDepth(),
        )
        np.testing.assert_allclose(
            msplot.points,
            np.array([
                [0.2112587, 3.0322570],
                [1.2823448, 0.8272850],
                [0.8646544, 1.8619370],
                [1.9862512, 5.5287354],
                [0.7534918, 0.7203502],
                [1.1325291, 0.2808455],
                [-2.650529, 0.9702889],
                [0.1434387, 0.9159834],
                [-0.402844, 0.6413531],
                [0.6354411, 0.6934311],
                [0.5727553, 0.4628254],
                [3.0524899, 8.8008899],
                [2.7355803, 10.338497],
                [3.1179374, 7.0686220],
                [3.4944047, 11.479432],
                [-0.402532, 0.5253690],
                [0.5782190, 5.5400704],
                [-0.839887, 0.7350041],
                [-3.456470, 1.1156415],
                [0.2260207, 1.5071672],
                [-0.561562, 0.8836978],
                [-1.690263, 0.6392155],
                [-0.385394, 0.7401909],
                [0.1467050, 0.9090058],
                [7.1811993, 39.003407],
                [6.8943132, 30.968126],
                [6.6227164, 41.448548],
                [0.0726709, 1.5960063],
                [1.4450617, 8.7183435],
                [-1.459836, 0.2719813],
                [-2.824349, 4.5729382],
                [-2.390462, 1.5464775],
                [-5.869571, 5.3517279],
                [-5.426019, 5.1817219],
                [-16.34459, 0.9397117],
            ]),
            rtol=1e-5,
        )
        np.testing.assert_array_almost_equal(
            msplot.outliers,
            np.array([  # noqa: WPS317
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                0, 0, 0, 0, 1,
            ]),
        )


if __name__ == '__main__':
    unittest.main()
