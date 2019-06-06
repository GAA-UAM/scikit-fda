import unittest
from skfda.representation.basis import Monomial, Fourier, FDataBasis
from skfda.ml.regression import LinearScalarRegression
import numpy as np

class TestRegression(unittest.TestCase):
    """Test regression"""

    def test_scalar_regression(self):
        beta_Basis = Fourier(nbasis=5)
        beta_fd = FDataBasis(beta_Basis, [1, 2, 3, 4, 5])

        x_Basis = Monomial(nbasis=7)
        x_fd = FDataBasis(x_Basis, np.identity(7))

        scalar_test = LinearScalarRegression([beta_fd])
        y = scalar_test.predict([x_fd])

        scalar = LinearScalarRegression([beta_Basis])
        scalar.fit([x_fd], y)
        np.testing.assert_array_almost_equal(scalar.beta[0].coefficients,
                                             beta_fd.coefficients)

        beta_Basis = Fourier(nbasis=5)
        beta_fd = FDataBasis(beta_Basis, [1, 1, 1, 1, 1])
        y = [1.0000684777229512,
             0.1623672257830915,
             0.08521053851548224,
             0.08514200869281137,
             0.09529138749665378,
             0.10549625973303875,
             0.11384314859153018]

        scalar = LinearScalarRegression([beta_Basis])
        scalar.fit([x_fd], y)
        np.testing.assert_array_almost_equal(scalar.beta[0].coefficients,
                                             beta_fd.coefficients)

if __name__ == '__main__':
    print()
    unittest.main()
