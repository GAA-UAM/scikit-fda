import unittest
from fda.basis import FDataBasis, Monomial
import numpy as np
from fda import math_basic
import scipy.stats.mstats


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_monomial_smoothing(self):
        # It does not have much sense to apply smoothing in this basic case
        # where the fit is very good but its just for testing purposes
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = Monomial(nbasis=4)
        fd = FDataBasis.from_data(x, t, basis,
                                  differential_operator=2,
                                  smoothing_factor=1)
        # These results where extracted from the R package fda
        np.testing.assert_array_equal(
            fd.coefficients.round(2), np.array([[ 0.61, -0.88,  0.06,  0.02]]))


if __name__ == '__main__':
    print()
    unittest.main()
