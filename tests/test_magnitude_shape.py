import unittest

import numpy as np
from skfda import FDataGrid
from skfda.exploratory.visualization.magnitude_shape_plot import (
    directional_outlyingness, MagnitudeShapePlot)
from skfda.datasets import fetch_weather


class TestMagnitudeShapePlot(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_directional_outlyingness(self):
        data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                       [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
                       [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        sample_points = [2, 4, 6, 8]
        fd = FDataGrid(data_matrix, sample_points)
        dir_outlyingness, mean_dir_outl, variation_dir_outl = directional_outlyingness(
            fd)
        np.testing.assert_allclose(dir_outlyingness,
                                   np.array([[[0., 0.],
                                              [0., 0.],
                                              [0., 0.],
                                              [0., 0.]],

                                             [[0.19611614, 0.03922323],
                                              [0.19611614, 0.03922323],
                                              [0.19611614, 0.03922323],
                                              [0.19900744, 0.01990074]],

                                             [[0.49937617, -0.02496881],
                                              [0.49937617, -0.02496881],
                                              [0.49937617, -0.02496881],
                                              [0.49937617, -0.02496881]]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(mean_dir_outl,
                                   np.array([[0., 0.],
                                             [0.29477656, 0.05480932],
                                             [0.74906425, -0.03745321]]),
                                   rtol=1e-06)
        np.testing.assert_allclose(variation_dir_outl,
                                   np.array([0., 0.01505136, 0.09375]))

    def test_magnitude_shape_plot(self):
        fd = fetch_weather()["data"]
        fd_temperatures = FDataGrid(data_matrix=fd.data_matrix[:, :, 0],
                                    sample_points=fd.sample_points,
                                    dataset_label=fd.dataset_label,
                                    axes_labels=fd.axes_labels[0:2])
        msplot = MagnitudeShapePlot(fd_temperatures, random_state=0)
        np.testing.assert_allclose(msplot.points,
                                   np.array([[0.28216472, 3.15069249],
                                             [1.43406267, 0.77729052],
                                             [0.96089808, 2.7302293],
                                             [2.1469911, 7.06601804],
                                             [0.89081951, 0.71098079],
                                             [1.22591999, 0.2363983],
                                             [-2.65530111, 0.9666511],
                                             [0.47819535, 0.83989187],
                                             [-0.11256072, 0.89035836],
                                             [0.99627103, 0.3255725],
                                             [0.77889317, 0.32451932],
                                             [3.47490723, 12.5630275],
                                             [3.14828582, 13.80605804],
                                             [3.51793514, 10.46943904],
                                             [3.94435195, 15.24142224],
                                             [0., 0.],
                                             [0.74574282, 6.68207165],
                                             [-0.82501844, 0.82694929],
                                             [-3.4617439, 1.10389229],
                                             [0.44523944, 1.61262494],
                                             [-0.52255157, 1.00486028],
                                             [-1.67260144, 0.74626351],
                                             [-0.10133788, 0.96326946],
                                             [0.36576472, 0.93071675],
                                             [7.57827303, 40.70985885],
                                             [7.51140842, 36.65641988],
                                             [7.13000185, 45.56574331],
                                             [0.28166597, 1.70861091],
                                             [1.55486533, 8.75149947],
                                             [-1.43363018, 0.36935927],
                                             [-2.79138743, 4.80007762],
                                             [-2.39987853, 1.54992208],
                                             [-5.87118328, 5.34300766],
                                             [-5.42854833, 5.1694065],
                                             [-16.34459211, 0.9397118]]))
        np.testing.assert_array_almost_equal(msplot.outliers,
                                             np.array(
                                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                                  0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
                                                  0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                                                  0, 0, 0, 0, 1]))


if __name__ == '__main__':
    print()
    unittest.main()
