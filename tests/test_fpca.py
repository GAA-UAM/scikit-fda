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

        # initialize weather data with only the temperature. Humidity not needed
        fd_data = fetch_weather_temp_only()
        n_basis = 8
        n_components = 4

        # initialize basis data
        basis = Fourier(n_basis=n_basis)
        fd_basis = fd_data.to_basis(basis)

        # pass functional principal component analysis to weather data
        fpca = FPCABasis(n_components)
        fpca.fit(fd_basis)

        # results obtained using Ramsay's R package
        results = [[0.9231551,  0.13649663,  0.35694509, 0.0092012, -0.0244525,
                    -0.02923873, -0.003566887, -0.009654571, -0.010006303],
                   [-0.3315211, -0.05086430,  0.89218521, 0.1669182,  0.2453900,
                    0.03548997,  0.037938051, -0.025777507,  0.008416904],
                   [-0.1379108,  0.91250892,  0.00142045, 0.2657423, -0.2146497,
                    0.16833314,  0.031509179, -0.006768189,  0.047306718],
                   [0.1247078,  0.01579953, -0.26498643, 0.4118705,  0.7617679,
                    0.24922635,  0.213305250, -0.180158701,  0.154863926]]
        results = np.array(results)

        # compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        for i in range(n_components):
            if np.sign(fpca.components.coefficients[i][0]) != np.sign(results[i][0]):
                results[i, :] *= -1
            for j in range(n_basis):
                self.assertAlmostEqual(fpca.components.coefficients[i][j],
                                       results[i][j],
                                       delta=0.03)


if __name__ == '__main__':
    unittest.main()
