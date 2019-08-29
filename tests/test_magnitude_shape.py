import unittest

import numpy as np
from skfda import FDataGrid
from skfda.datasets import fetch_weather
from skfda.exploratory.depth import modified_band_depth
from skfda.exploratory.visualization import MagnitudeShapePlot


class TestMagnitudeShapePlot(unittest.TestCase):

    def test_magnitude_shape_plot(self):
        fd = fetch_weather()["data"]
        fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0],
                                    sample_points=fd.sample_points,
                                    dataset_label=fd.dataset_label,
                                    axes_labels=fd.axes_labels[0:2])
        msplot = MagnitudeShapePlot(
            fd_temperatures, depth_method=modified_band_depth)
        np.testing.assert_allclose(msplot.points,
                                   np.array([[0.2591055,   3.15861149],
                                             [1.3811996,   0.91806814],
                                             [0.94648379,   2.75695426],
                                             [2.11346208,   7.24045853],
                                             [0.82557436,   0.82727771],
                                             [1.23249759,   0.22004329],
                                             [-2.66259589,   0.96925352],
                                             [0.15827963,   1.00235557],
                                             [-0.43751765,   0.66236714],
                                             [0.70695162,   0.66482897],
                                             [0.73095525,   0.33165117],
                                             [3.48445368,  12.59745018],
                                             [3.15539264,  13.85234879],
                                             [3.52759979,  10.49810783],
                                             [3.95518808,  15.28317686],
                                             [-0.48486514,   0.5035343],
                                             [0.64492781,   6.83385521],
                                             [-0.83185751,   0.81125541],
                                             [-3.47125418,   1.10683451],
                                             [0.22241054,   1.76783493],
                                             [-0.54402406,   0.95229119],
                                             [-1.71310618,   0.61875513],
                                             [-0.44161441,   0.77815135],
                                             [0.13851408,   1.02560672],
                                             [7.59909246,  40.82126568],
                                             [7.57868277,  36.03923856],
                                             [7.12930634,  45.96866318],
                                             [0.05746528,   1.75817588],
                                             [1.53092075,   8.85227],
                                             [-1.48696387,   0.22472872],
                                             [-2.853082,   4.50814844],
                                             [-2.42297615,   1.46926902],
                                             [-5.8873129,   5.35742609],
                                             [-5.44346193,   5.18338576],
                                             [-16.38949483,   0.94027717]]
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
