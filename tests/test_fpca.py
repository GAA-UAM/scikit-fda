import unittest

import numpy as np
from skfda import FDataGrid
from skfda.exploratory.fpca import FPCABasis, FPCADiscretized
from skfda.datasets import fetch_growth, fetch_weather


def fetch_weather_temp_only():
    weather_dataset = fetch_weather()
    fd_data = weather_dataset['data']
    fd_data.data_matrix = fd_data.data_matrix[:, :, :1]
    fd_data.axes_labels = fd_data.axes_labels[:-1]
    return fd_data

class MyTestCase(unittest.TestCase):
    def test_basis_fpca_fit(self):
        fpca = FPCABasis()
        with self.assertRaises(AttributeError):
            fpca.fit(None)




if __name__ == '__main__':
    unittest.main()
