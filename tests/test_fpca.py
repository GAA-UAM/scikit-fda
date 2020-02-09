import unittest

import numpy as np
from skfda import FDataGrid, FDataBasis
from skfda.representation.basis import Fourier
from skfda.exploratory.fpca import FPCABasis, FPCADiscretized
from skfda.datasets import fetch_weather


def fetch_weather_temp_only():
    weather_dataset = fetch_weather()
    fd_data = weather_dataset['data']
    fd_data.data_matrix = fd_data.data_matrix[:, :, :1]
    fd_data.axes_labels = fd_data.axes_labels[:-1]
    return fd_data

class MyTestCase(unittest.TestCase):

    def test_basis_fpca_fit_attributes(self):
        fpca = FPCABasis()
        with self.assertRaises(AttributeError):
            fpca.fit(None)

        basis = Fourier(n_basis=1)
        # check that if n_components is bigger than the number of samples then
        # an exception should be thrown
        fd = FDataBasis(basis, [[0.9]])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

        # check that n_components must be smaller than the number of elements
        # of target basis
        fd = FDataBasis(basis, [[0.9], [0.7], [0.5]])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

    def test_discretized_fpca_fit_attributes(self):
        fpca = FPCADiscretized()
        with self.assertRaises(AttributeError):
            fpca.fit(None)

        # check that if n_components is bigger than the number of samples then
        # an exception should be thrown
        fd = FDataGrid([[0.5], [0.1]], sample_points=[0])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

        # check that n_components must be smaller than the number of attributes
        # in the FDataGrid object
        fd = FDataGrid([[0.9], [0.7], [0.5]], sample_points=[0])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

    def test_basis_fpca_fit_result(self):

        n_basis = 3
        n_components = 2

        # initialize basis data
        basis = Fourier(n_basis=n_basis)
        fd_basis = FDataBasis(basis,
                              [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0],
                               [0.0, 0.0, 3.0]])
        # pass functional principal component analysis to weather data
        fpca = FPCABasis(n_components)
        fpca.fit(fd_basis)

        # results obtained using Ramsay's R package
        results = [[-0.1010156, -0.4040594, 0.9091380],
                   [-0.5050764,  0.8081226, 0.3030441]]
        results = np.array(results)

        # compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        for i in range(n_components):
            if np.sign(fpca.components.coefficients[i][0]) != np.sign(results[i][0]):
                results[i, :] *= -1
            for j in range(n_basis):
                self.assertAlmostEqual(fpca.components.coefficients[i][j],
                                       results[i][j], delta=0.00001)


if __name__ == '__main__':
    unittest.main()
