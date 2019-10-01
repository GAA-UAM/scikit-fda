import unittest

import numpy as np
from skfda.ml.regression import LinearScalarRegression
from skfda.representation.basis import (FDataBasis, Constant, Monomial,
                                        Fourier,  BSpline)


class TestLinearScalarRegression(unittest.TestCase):

    def test_regression_fit(self):

        x_basis = Monomial(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(n_basis=5)
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
        np.testing.assert_array_almost_equal(scalar.beta_[0].coefficients,
                                             beta_fd.coefficients)

    def test_regression_predict_single_explanatory(self):

        x_basis = Monomial(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(n_basis=5)
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
        np.testing.assert_array_almost_equal(scalar.beta_[0].coefficients,
                                             beta_fd.coefficients)

    def test_regression_predict_multiple_explanatory(self):
        y = [1, 2, 3, 4, 5, 6, 7]

        x0 = FDataBasis(Constant(domain_range=(0, 1)), np.ones((7, 1)))
        x1 = FDataBasis(Monomial(n_basis=7), np.identity(7))

        beta0 = Constant(domain_range=(0, 1))
        beta1 = BSpline(domain_range=(0, 1), n_basis=5)

        scalar = LinearScalarRegression([beta0, beta1])

        scalar.fit([x0, x1], y)

        betas = scalar.beta_

        np.testing.assert_array_almost_equal(betas[0].coefficients.round(4),
                                             np.array([[32.6518]]))

        np.testing.assert_array_almost_equal(betas[1].coefficients.round(4),
                                             np.array([[-28.6443,
                                                        80.3996,
                                                        -188.587,
                                                        236.5832,
                                                        -481.3449]]))

    def test_error_X_not_FData(self):
        """Tests that at least one of the explanatory variables
        is an FData object. """

        x_fd = np.identity(7)
        y = np.zeros(7)

        scalar = LinearScalarRegression([Fourier(n_basis=5)])

        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_y_is_FData(self):
        """Tests that none of the explained variables is an FData object
        """
        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = list(FDataBasis(Monomial(n_basis=7), np.identity(7)))

        scalar = LinearScalarRegression([Fourier(n_basis=5)])

        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_X_beta_len_distinct(self):
        """ Test that the number of beta bases and explanatory variables
        are not different """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = Fourier(n_basis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd, x_fd], y)

        scalar = LinearScalarRegression([beta, beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_y_X_samples_different(self):
        """ Test that the number of response samples and explanatory samples
        are not different """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(8)]
        beta = Fourier(n_basis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

        x_fd = FDataBasis(Monomial(n_basis=8), np.identity(8))
        y = [1 for _ in range(7)]
        beta = Fourier(n_basis=5)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_beta_not_basis(self):
        """ Test that all beta are Basis objects. """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = FDataBasis(Monomial(n_basis=7), np.identity(7))

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y)

    def test_error_weights_lenght(self):
        """ Test that the number of weights is equal to the
        number of samples """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [1 for _ in range(8)]
        beta = Monomial(n_basis=7)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y, weights)

    def test_error_weights_negative(self):
        """ Test that none of the weights are negative. """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [-1 for _ in range(7)]
        beta = Monomial(n_basis=7)

        scalar = LinearScalarRegression([beta])
        np.testing.assert_raises(ValueError, scalar.fit, [x_fd], y, weights)


if __name__ == '__main__':
    print()
    unittest.main()
