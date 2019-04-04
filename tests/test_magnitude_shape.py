import unittest

import numpy as np
from fda.grid import FDataGrid
from fda.magnitude_shape_plot import directional_outlyingness, magnitude_shape_plot
from fda.datasets import fetch_weather


class TestMagnitudeShapePlot(unittest.TestCase):

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

    def test_magnitude_shape_plot(self):
        fd = fetch_weather()["data"]
        fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0], sample_points=fd.sample_points,
                                    dataset_label=fd.dataset_label, axes_labels=fd.axes_labels[0:2])
        points, outliers = magnitude_shape_plot(fd_temperatures)
        np.testing.assert_allclose(points,
                                   np.array([[0.29605473, 3.1973174],
                                             [1.42918812, 0.77167217],
                                             [0.95067136, 2.52426754],
                                             [2.16992644, 7.19307267],
                                             [0.88398291, 0.71342667],
                                             [1.2338771, 0.26077806],
                                             [-2.66545805, 0.98224958],
                                             [0.47485269, 0.85015947],
                                             [-0.11915288, 0.89040602],
                                             [1.00493706, 0.32724658],
                                             [0.78370207, 0.32590353],
                                             [3.51020498, 13.23706999],
                                             [3.14731034, 13.71266401],
                                             [3.56782288, 11.25061776],
                                             [3.89916762, 14.59791781],
                                             [0., 0.],
                                             [0.69956016, 5.94263925],
                                             [-0.82783633, 0.82218997],
                                             [-3.47972584, 1.12757531],
                                             [0.44656982, 1.67994682],
                                             [-0.53854927, 0.99742639],
                                             [-1.67871103, 0.72462732],
                                             [-0.09859915, 0.96533453],
                                             [0.36566853, 0.94215759],
                                             [7.61887747, 40.8437447],
                                             [7.46386818, 35.95988241],
                                             [7.27510048, 46.6765394],
                                             [0.27975955, 1.69430826],
                                             [1.56583899, 8.75429748],
                                             [-1.43804126, 0.37016268],
                                             [-2.80873277, 4.80759694],
                                             [-2.39978195, 1.50862696],
                                             [-5.87192676, 5.37450281],
                                             [-5.45783514, 5.42086578],
                                             [-16.38192599, 1.00378603]]))
        np.testing.assert_array_equal(outliers,
                                      np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
                                                1.]))


if __name__ == '__main__':
    print()
    unittest.main()
