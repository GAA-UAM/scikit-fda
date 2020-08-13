from skfda import FDataGrid
from skfda.datasets import fetch_weather
from skfda.exploratory.depth import modified_band_depth
from skfda.exploratory.visualization import MagnitudeShapePlot
import unittest

import numpy as np


class TestMagnitudeShapePlot(unittest.TestCase):

    def test_magnitude_shape_plot(self):
        fd = fetch_weather()["data"]
        fd_temperatures = fd.coordinates[0]
        msplot = MagnitudeShapePlot(
            fd_temperatures, depth_method=modified_band_depth)
        np.testing.assert_allclose(msplot.points,
                                   np.array([[0.25839562,   3.14995827],
                                             [1.3774155,   0.91556716],
                                             [0.94389069,   2.74940766],
                                             [2.10767177,   7.22065509],
                                             [0.82331252,   0.8250163],
                                             [1.22912089,   0.2194518],
                                             [-2.65530111,   0.9666511],
                                             [0.15784599,   0.99960958],
                                             [-0.43631897,   0.66055387],
                                             [0.70501476,   0.66301126],
                                             [0.72895263,   0.33074653],
                                             [3.47490723,  12.5630275],
                                             [3.14674773,  13.81447167],
                                             [3.51793514,  10.46943904],
                                             [3.94435195,  15.24142224],
                                             [-0.48353674,   0.50215652],
                                             [0.64316089,   6.81513544],
                                             [-0.82957845,   0.80903798],
                                             [-3.4617439,   1.10389229],
                                             [0.2218012,   1.76299192],
                                             [-0.54253359,   0.94968438],
                                             [-1.70841274,   0.61708188],
                                             [-0.44040451,   0.77602089],
                                             [0.13813459,   1.02279698],
                                             [7.57827303,  40.70985885],
                                             [7.55791925,  35.94093086],
                                             [7.10977399,  45.84310211],
                                             [0.05730784,   1.75335899],
                                             [1.52672644,   8.82803475],
                                             [-1.48288999,   0.22412958],
                                             [-2.84526533,   4.49585828],
                                             [-2.41633786,   1.46528758],
                                             [-5.87118328,   5.34300766],
                                             [-5.42854833,   5.1694065],
                                             [-16.34459211,   0.9397118]]
                                            ))
        np.testing.assert_array_almost_equal(msplot.outliers,
                                             np.array(
                                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                                  0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
                                                  0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                                                  0, 0, 0, 0, 1]))


if __name__ == '__main__':
    print()
    unittest.main()
