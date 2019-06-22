import unittest
from skfda.representation.basis import Monomial, Fourier, FDataBasis
from skfda.ml.regression import LinearScalarRegression
import numpy as np


class TestLinearScalarRegression(unittest.TestCase):

    def test_regression_fit(self):

        x_basis = Monomial(nbasis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(nbasis=5)
        beta_fd = FDataBasis(beta_basis, [1, 1, 1, 1, 1])
        y = [1.0000684777229512,
             0.1623672257830915,
             0.08521053851548224,
             0.08514200869281137,
             0.09529138749665378,
             0.10549625973303875,
             0.11384314859153018]

        scalar = LinearScalarRegression([beta_basis])
        scalar.fit([x_fd], y)
        np.testing.assert_array_almost_equal(scalar.beta[0].coefficients,
                                             beta_fd.coefficients)


    def test_regression_predict(self):

        x_basis = Monomial(nbasis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(nbasis=5)
        beta_fd = FDataBasis(beta_basis, [1, 1, 1, 1, 1])
        y = [1.0000684777229512,
             0.1623672257830915,
             0.08521053851548224,
             0.08514200869281137,
             0.09529138749665378,
             0.10549625973303875,
             0.11384314859153018]

        scalar = LinearScalarRegression([beta_basis])
        scalar.fit([x_fd], y)
        np.testing.assert_array_almost_equal(scalar.beta[0].coefficients,
                                             beta_fd.coefficients)

    def test_error_X_not_FData(self):
        """Tests that at least one of the explanatory variables
        is an FData object. """

        x_fd = np.identity(7)
        y = np.zeros(7)

        scalar = LinearScalarRegression([Fourier(nbasis=5)])

        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_y_is_FData(self):
        """Tests that none of the explained variables is an FData object
        """
        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = list(FDataBasis(Monomial(nbasis=7), np.identity(7)))

        scalar = LinearScalarRegression([Fourier(nbasis=5)])

        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_X_beta_len_distinct(self):
        """ Test that the number of beta bases and explanatory variables
        are not different """

        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = Fourier(nbasis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd, x_fd], y)

        scalar = LinearScalarRegression([beta, beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_y_X_samples_different(self):
        """ Test that the number of response samples and explanatory samples
        are not different """

        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = [1 for _ in range(8)]
        beta = Fourier(nbasis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

        x_fd = FDataBasis(Monomial(nbasis=8), np.identity(8))
        y = [1 for _ in range(7)]
        beta = Fourier(nbasis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_beta_not_basis(self):
        """ Test that all beta are Basis objects. """

        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = FDataBasis(Monomial(nbasis=7), np.identity(7))

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_weights_lenght(self):
        """ Test that the number of weights is equal to the
        number of samples """

        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [1 for _ in range(8)]
        beta = Monomial(nbasis=7)

        scalar = LinearScalarRegression([beta], weights)
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_weights_negative(self):
        """ Test that none of the weights are negative. """

        x_fd = FDataBasis(Monomial(nbasis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [-1 for _ in range(7)]
        beta = Monomial(nbasis=7)

        scalar = LinearScalarRegression([beta], weights)
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)


if __name__ == '__main__':
    print()
    unittest.main()
