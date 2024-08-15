"""Tests for scalar and functional response linear regression."""
from __future__ import annotations

import unittest
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from sklearn.preprocessing import OneHotEncoder

from skfda.datasets import fetch_weather, make_gaussian, make_gaussian_process
from skfda.misc.covariances import Gaussian
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import HistoricalLinearRegression, LinearRegression
from skfda.representation.basis import (
    BSplineBasis,
    ConstantBasis,
    FDataBasis,
    FourierBasis,
    MonomialBasis,
)
from skfda.representation.grid import FDataGrid


def _test_linear_regression_common(
    X_train,
    y_train,
    expected_coefs,
    X_test,
    y_test,
    coef_basis=None,
    regularization=None,
) -> None:
    """Execute a test of linear regression, given the parameters."""
    linear_regression = LinearRegression(
        fit_intercept=False,
        coef_basis=coef_basis,
        regularization=regularization,
    )
    linear_regression.fit(X_train, y_train)

    for coef, expected in zip(linear_regression.coef_, expected_coefs):
        assert isinstance(coef, FDataBasis)
        assert coef.basis == expected.basis
        np.testing.assert_allclose(
            coef.coefficients,
            expected.coefficients,
            atol=1e-6,
        )

    y_pred = linear_regression.predict(X_test)
    assert isinstance(y_pred, FDataBasis)
    assert y_pred.basis == y_test.basis
    np.testing.assert_allclose(
        y_pred.coefficients,
        y_test.coefficients,
        rtol=1e-5,
    )


class TestScalarLinearRegression(unittest.TestCase):
    """Tests for linear regression with scalar response."""

    def test_single_explanatory(self) -> None:
        """Test a basic example of functional regression.

        Scalar response with functional covariates.
        """
        x_basis = MonomialBasis(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = FourierBasis(n_basis=5)
        beta_fd = FDataBasis(beta_basis, [1, 1, 1, 1, 1])
        y = [
            0.9999999999999993,
            0.162381381441085,
            0.08527083481359901,
            0.08519946930844623,
            0.09532291032042489,
            0.10550022969639987,
            0.11382675064746171,
        ]

        scalar = LinearRegression(coef_basis=[beta_basis])
        scalar.fit(x_fd, y)
        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            beta_fd.coefficients,
        )
        np.testing.assert_allclose(
            scalar.intercept_, 0.0, atol=1e-6,
        )

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

        scalar = LinearRegression(
            coef_basis=[beta_basis],
            fit_intercept=False,
        )
        scalar.fit(x_fd, y)
        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            beta_fd.coefficients,
        )
        np.testing.assert_equal(
            scalar.intercept_, 0.0,
        )

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

    def test_multiple_explanatory(self) -> None:
        """Test a example of functional regression.

        Scalar response with functional covariates.
        """
        y = [1, 2, 3, 4, 5, 6, 7]

        X = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))

        beta1 = BSplineBasis(domain_range=(0, 1), n_basis=5)

        scalar = LinearRegression(coef_basis=[beta1])

        scalar.fit(X, y)

        np.testing.assert_allclose(
            scalar.intercept_.round(4), np.array([32.65]), rtol=1e-3,
        )

        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients.round(4),
            np.array([[
                -28.6443,
                80.3996,
                -188.587,
                236.5832,
                -481.3449,
            ]]), rtol=1e-3,
        )

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_mixed(self) -> None:
        """Test a example of functional regression.

        Scalar response with multivariate and functional covariates.
        """
        multivariate = np.array(
            [[0, 0], [2, 7], [1, 7], [3, 9], [4, 16], [2, 14], [3, 5]],
        )

        x_fd = FDataBasis(MonomialBasis(n_basis=3), [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        X = [multivariate, x_fd]

        # y = 2 + sum([3, 1] * array) + int(3 * function)  # noqa: E800
        intercept = 2
        coefs_multivariate = np.array([3, 1])
        coefs_functions = FDataBasis(
            MonomialBasis(n_basis=3), [[3, 0, 0]],
        )
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression()
        scalar.fit(X, y)

        np.testing.assert_allclose(
            scalar.intercept_, intercept, atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[0], coefs_multivariate, atol=0.01,
        )

        assert isinstance(scalar.coef_[1], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[1].coefficients,
            coefs_functions.coefficients,
            atol=0.01,
        )

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_same_result_1d_2d_multivariate_arrays(self) -> None:
        """Test if the results using 1D and 2D arrays are the same.

        1D and 2D multivariate arrays are allowed in LinearRegression
        interface, and the result must be the same.
        """
        multivariate1 = np.asarray([1, 2, 4, 1, 3, 2])
        multivariate2 = np.asarray([7, 3, 2, 1, 1, 5])
        multivariate = [[1, 7], [2, 3], [4, 2], [1, 1], [3, 1], [2, 5]]

        x_basis = MonomialBasis(n_basis=2)
        x_fd = FDataBasis(
            x_basis,
            [[0, 2], [0, 4], [1, 0], [2, 0], [1, 2], [2, 2]],
        )

        y = [11, 10, 12, 6, 10, 13]

        linear = LinearRegression(coef_basis=[None, ConstantBasis()])
        linear.fit([multivariate, x_fd], y)

        linear2 = LinearRegression(coef_basis=[None, None, ConstantBasis()])
        linear2.fit([multivariate1, multivariate2, x_fd], y)

        np.testing.assert_equal(
            linear.coef_[0][0],
            linear2.coef_[0],
        )

        np.testing.assert_equal(
            linear.coef_[0][1],
            linear2.coef_[1],
        )

        np.testing.assert_equal(
            linear.coef_[1],
            linear2.coef_[2],
        )

    def test_df_multivariate(self) -> None:
        """Test a example of functional regression with Dataframe input.

        Scalar response with multivariate and functional covariates.
        """
        multivariate1 = [0, 2, 1, 3, 4, 2, 3]
        multivariate2 = [0, 7, 7, 9, 16, 14, 5]
        multivariate = [list(obs) for obs in zip(multivariate1, multivariate2)]

        x_basis = MonomialBasis(n_basis=3)
        x_fd = FDataBasis(x_basis, [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        cov_dict = {"fd": x_fd, "mult1": multivariate1, "mult2": multivariate2}

        df = pd.DataFrame(cov_dict)

        # y = 2 + sum([3, 1] * array) + int(3 * function)  # noqa: E800
        intercept = 2
        coefs_multivariate = np.array([3, 1])
        coefs_functions = FDataBasis(
            MonomialBasis(n_basis=3), [[3, 0, 0]],
        )
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression()
        scalar.fit(df, y)

        np.testing.assert_allclose(
            scalar.intercept_, intercept, atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[1], coefs_multivariate[0], atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[2], coefs_multivariate[1], atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            coefs_functions.coefficients,
            atol=0.01,
        )

        y_pred = scalar.predict(df)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_mixed_regularization(self) -> None:
        """Test a example of functional regression.

        Scalar response with multivariate and functional covariates
        using regularization.
        """
        multivariate = np.array([
            [0, 0], [2, 7], [1, 7], [3, 9], [4, 16], [2, 14], [3, 5],
        ])

        x_fd = FDataBasis(MonomialBasis(n_basis=3), [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        X: Sequence[
            np.typing.NDArray[np.float64] | FDataBasis,
        ] = [multivariate, x_fd]

        # y = 2 + sum([3, 1] * array) + int(3 * function)  # noqa: E800
        intercept = 2
        coefs_multivariate = np.array([3, 1])
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression(
            regularization=[
                L2Regularization(lambda x: x),
                L2Regularization(
                    LinearDifferentialOperator(2),
                ),
            ],
        )
        scalar.fit(X, y)

        np.testing.assert_allclose(
            scalar.intercept_, intercept, atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[0],
            [2.536739, 1.072186],
            atol=0.01,
        )

        assert isinstance(scalar.coef_[1], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[1].coefficients,
            [[2.125676, 2.450782, 5.808745e-4]],
            atol=0.01,
        )

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(
            y_pred,
            [
                5.349035,
                16.456464,
                13.361185,
                23.930295,
                32.650965,
                23.961766,
                16.29029,
            ],
            atol=0.01,
        )

    def test_regularization(self) -> None:
        """Test a example of functional regression.

        Scalar response with functional covariates using regularization.
        """
        x_basis = MonomialBasis(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = FourierBasis(n_basis=5)
        beta_fd = FDataBasis(beta_basis, [1.0403, 0, 0, 0, 0])
        y = [
            1.0000684777229512,
            0.1623672257830915,
            0.08521053851548224,
            0.08514200869281137,
            0.09529138749665378,
            0.10549625973303875,
            0.11384314859153018,
        ]

        y_pred_compare = [
            0.890341,
            0.370162,
            0.196773,
            0.110079,
            0.058063,
            0.023385,
            -0.001384,
        ]

        scalar = LinearRegression(
            coef_basis=[beta_basis],
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
        )
        scalar.fit(x_fd, y)
        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            beta_fd.coefficients,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            scalar.intercept_, -0.15, atol=1e-4,
        )

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_pred_compare, atol=1e-4)

        x_basis = MonomialBasis(n_basis=3)
        x_fd = FDataBasis(x_basis, [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
        ])

        beta_fd = FDataBasis(x_basis, [3, 2, 1])
        y = np.array([1 + 13 / 3, 1 + 29 / 12, 1 + 17 / 10, 1 + 311 / 30])

        # Non regularized
        scalar = LinearRegression()
        scalar.fit(x_fd, y)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients, beta_fd.coefficients,
        )
        np.testing.assert_allclose(scalar.intercept_, 1)

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

        # Regularized
        beta_fd_reg = FDataBasis(x_basis, [2.812, 3.043, 0])
        y_reg = [5.333, 3.419, 2.697, 11.366]

        scalar_reg = LinearRegression(
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
        )
        scalar_reg.fit(x_fd, y)
        assert isinstance(scalar_reg.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar_reg.coef_[0].coefficients,
            beta_fd_reg.coefficients,
            atol=0.001,
        )
        np.testing.assert_allclose(
            scalar_reg.intercept_, 0.998, atol=0.001,
        )

        y_pred = scalar_reg.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_reg, atol=0.001)

    def test_error_y_X_samples_different(self) -> None:  # noqa: N802
        """Number of response samples and explanatory samples are not different.

        Raises ValueError when response is scalar.
        """
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = [1 for _ in range(8)]
        beta = FourierBasis(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_beta_not_basis(self) -> None:
        """Test that all beta are Basis objects."""
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(7)])
        beta = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(TypeError):
            scalar.fit([x_fd], y)

    def test_error_weights_lenght(self) -> None:
        """Number of weights is equal to the number of samples."""
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(7)])
        weights = np.array([1 for _ in range(8)])
        beta = MonomialBasis(n_basis=7)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y, weights)

    def test_error_weights_negative(self) -> None:
        """Test that none of the weights are negative."""
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(7)])
        weights = np.array([-1 for _ in range(7)])
        beta = MonomialBasis(n_basis=7)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y, weights)


class TestFunctionalLinearRegression(unittest.TestCase):
    """Tests for linear regression with functional response."""

    def test_multivariate_covariates_constant_basic(self) -> None:
        """
        Univariate with functional response and constant coefficient one.
        """
        y_basis = ConstantBasis()

        X_train = pd.DataFrame({
            "covariate": [1, 3, 5],
        })
        y_train = FDataBasis(
            basis=y_basis,
            coefficients=[
                [1],
                [3],
                [5],
            ],
        )

        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[[1]],
            )
        ]

        X_test = pd.DataFrame({
            "covariate": [2, 4, 6],
        })
        y_test = FDataBasis(
            basis=y_basis,
            coefficients=[
                [2],
                [4],
                [6],
            ],
        )

        _test_linear_regression_common(
            X_train=X_train,
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test,
            y_test=y_test,
        )

        # Check also without dataframes
        _test_linear_regression_common(
            X_train=X_train.to_numpy(),
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test.to_numpy(),
            y_test=y_test,
        )

    def test_multivariate_covariates_monomial_basic(self) -> None:
        """
        Multivariate with functional response and identity coefficients.
        """
        y_basis = MonomialBasis(n_basis=2)

        X_train = pd.DataFrame({
            "covariate_1": [1, 3, 5],
            "covariate_2": [2, 4, 6],
        })
        y_train = FDataBasis(
            basis=y_basis,
            coefficients=[
                [1, 2],
                [3, 4],
                [5, 6],
            ],
        )

        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[[1, 0]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[0, 1]],
            ),
        ]

        X_test = pd.DataFrame({
            "covariate_1": [2, 4, 6],
            "covariate_2": [1, 3, 5],
        })
        y_test = FDataBasis(
            basis=y_basis,
            coefficients=[
                [2, 1],
                [4, 3],
                [6, 5],
            ],
        )

        _test_linear_regression_common(
            X_train=X_train,
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test,
            y_test=y_test,
        )

        # Check also without dataframes
        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[
                    [1, 0],
                    [0, 1],
                ],
            ),
        ]

        # Currently broken.

        # _test_linear_regression_common(
        #     X_train=X_train.to_numpy(),
        #     y_train=y_train,
        #     expected_coefs=expected_coefs,
        #     X_test=X_test.to_numpy(),
        #     y_test=y_test,
        # )

    def test_multivariate_3_covariates(self) -> None:
        """Test a more complex example involving 3 covariates."""
        y_basis = MonomialBasis(n_basis=3)

        X_train = pd.DataFrame({
            "covariate_1": [3, 5, 3],
            "covariate_2": [4, 1, 2],
            "covariate_3": [1, 6, 8],
        })
        y_train = FDataBasis(
            basis=y_basis,
            coefficients=[
                [47, 22, 24],
                [43, 47, 39],
                [40, 53, 51],
            ],
        )

        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[[6, 3, 1]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[7, 2, 4]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[1, 5, 5]],
            ),
        ]

        X_test = pd.DataFrame({
            "covariate_1": [3],
            "covariate_2": [2],
            "covariate_3": [1],
        })
        y_test = FDataBasis(
            basis=y_basis,
            coefficients=[[33, 18, 16]],
        )

        _test_linear_regression_common(
            X_train=X_train,
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test,
            y_test=y_test,
        )

    def test_multivariate_covariates_regularization(self) -> None:
        """Test a example of functional regression.

        Functional response with multivariate covariates and
        beta regularization.
        """
        y_basis = MonomialBasis(n_basis=3)

        X_train = pd.DataFrame({
            "covariate_1": [3, 5, 3],
            "covariate_2": [4, 1, 2],
            "covariate_3": [1, 6, 8],
        })
        y_train = FDataBasis(
            basis=y_basis,
            coefficients=[
                [47, 22, 24],
                [43, 47, 39],
                [40, 53, 51],
            ],
        )

        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[[5.769441, 3.025921, 1.440655]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[6.688267, 1.938523, 3.579894]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[1.198499, 4.952166, 4.811818]],
            ),
        ]

        X_test = pd.DataFrame({
            "covariate_1": [3],
            "covariate_2": [2],
            "covariate_3": [1],
        })
        y_test = FDataBasis(
            basis=y_basis,
            coefficients=[[31.883356, 17.906975, 16.293571]],
        )

        _test_linear_regression_common(
            X_train=X_train,
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test,
            y_test=y_test,
            regularization=[L2Regularization()] * 3,
        )

    def test_multivariate_covariates_R_fda(self) -> None:  # noqa: N802
        """Test a example with Canadian Weather comparing with R fda package.

        Code used in R:
            daybasis65 <- create.fourier.basis(
                rangeval=c(0, 365), nbasis=65, axes=list('axesIntervals'))
            Temp.fd <- with(CanadianWeather, smooth.basisPar(day.5,
                             dailyAv[,,'Temperature.C'], daybasis65)$fd)
            TempRgn.f <- fRegress(Temp.fd ~ region, CanadianWeather)
            write.table(
                t(round(
                    TempRgn.f$betaestlist$const$fd$coefs,
                    digits=4)),
                file="", sep = ",", col.names = FALSE, row.names = FALSE
            )
            write.table(
                t(round(
                    TempRgn.f$betaestlist$region.Atlantic$fd$coefs,
                    digits=4)),
                file="", sep = ",", col.names = FALSE, row.names = FALSE
            )
            write.table(
                t(round(
                    TempRgn.f$betaestlist$region.Continental$fd$coefs,
                    digits=4)),
                file="", sep = ",", col.names = FALSE, row.names = FALSE)
            write.table(
                t(round(
                    TempRgn.f$betaestlist$region.Pacific$fd$coefs,
                    digits=4)),
                file="", sep = ",", col.names = FALSE, row.names = FALSE)
        """
        X_weather, y_weather = fetch_weather(
            return_X_y=True, as_frame=True,
        )
        fd = X_weather.iloc[:, 0].values

        y_basis = FourierBasis(n_basis=65)
        y_fd = fd.coordinates[0].to_basis(y_basis)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit([['Atlantic'], ['Continental'], ['Pacific']])
        X = np.array(y_weather).reshape(-1, 1)
        X = enc.transform(X).toarray()

        cov_dict = {"mult1": X[:, 0], "mult2": X[:, 1], "mult3": X[:, 2]}
        df = pd.DataFrame(cov_dict)

        beta_const_R = [  # noqa: WPS317
            -225.5085, -110.817, -243.4708, 4.6815, 21.4488, 10.3784, 2.6317,
            1.7571, 2.4812, -1.5179, 1.4451, -0.6581, 2.8287, 0.4106, 1.5839,
            -1.711, 0.5587, -2.2268, 2.4745, -0.5179, -0.8376, -3.1504,
            -0.1357, -0.1186, 1.1512, 0.7343, 1.842, -0.5258, 1.2263, -0.576,
            -0.6754, -0.6952, -0.416, -1.0292, 1.6742, 0.4276, 0.5185, -0.2135,
            0.3239, 1.6598, 1.0682, 2.2478, 0.2692, 1.8589, -0.5416, 0.5256,
            -1.6838, -1.1174, 0.1842, -0.3521, 0.1809, -1.6302, 0.6676,
            -0.3356, 1.036, -0.6297, 0.4227, -0.3096, 1.1373, 0.6317, 0.3608,
            -0.9949, -0.709, -0.4588, -0.5694,
        ]

        beta_atlantic_R = [  # noqa: WPS317
            312.966, 35.9273, 67.7156, -12.9111, -27.3945, -18.3422,
            -6.6074, -0.0203, -4.5716, 3.3142, -1.8419, 2.2008, -3.1554,
            -0.8167, -1.6248, 1.4791, -0.8676, 2.9854, -2.5819, -0.239, 0.6418,
            2.2211, 1.4992, -2.2746, 0.6767, -2.8692, 1.478, 0.5988, -0.3434,
            -0.2574, 2.3693, -0.016, 1.4911, 3.2798, -0.6508, 1.3326, -0.6729,
            1.0736, -0.7874, -1.2653, -1.8837, -3.1971, 0.0166, -1.298, 0.1403,
            -1.2479, 0.593, 0.715, 0.1659, 0.8047, -1.2938, 0.7217, -1.1323,
            -0.9719, -1.256, 0.8089, -0.1986, 0.7974, -0.4129, -0.6855,
            -0.6397, 3.2471, 0.4686, 1.3593, 0.9434,
        ]

        beta_continental_R = [  # noqa: WPS317
            214.8319, 41.1702, 6.2763, -11.5837, -40.6003, -10.9865, -6.6548,
            4.2589, -3.5174, 0.9494, 1.5624, -3.1435, -1.3242, -1.6431,
            -1.0234, 2.0606, -1.1042, -0.1723, -4.2717, -0.9321, 1.2331,
            2.0911, -1.0444, -1.757, -1.9564, -2.3117, -3.0405, -1.3801,
            -1.7431, -2.0031, 0.7171, -0.6877, 0.7969, -1.01, -0.1761, -2.7614,
            0.8308, -0.7232, 1.671, 0.0118, 1.8239, 0.5399, 1.8575, 0.9313,
            1.6813, 0.834, 2.1028, 1.8707, -0.147, -0.6401, -0.165, 1.5439,
            -0.4666, 0.2153, -0.8795, 0.4695, 0.0417, 0.7045, -1.1045, 0.0166,
            -0.7447, 1.4645, 1.5654, -0.3106, 0.7647,
        ]

        beta_pacific_R = [  # noqa: WPS317
            375.1732, 78.6384, 127.8782, 6.0014, -29.3124, -11.4446, -5.3623,
            -1.1054, -5.4936, 0.5137, 0.0086, -0.7174, -5.2713, -1.2635,
            -1.6654, -0.5359, -2.4626, 1.8152, -4.0212, 0.8431, -1.7737,
            3.7342, -2.0556, 0.0382, -2.4436, -1.9431, -3.6757, -0.6956,
            -2.8307, -0.7396, 1.6465, -0.3534, 0.903, 0.0484, -1.6763, -1.6237,
            -0.9657, -1.6763, -0.2481, -1.3371, -0.6295, -2.4142, 0.9318,
            -1.1531, 0.8854, -0.966, 1.6884, 1.6327, -0.1843, 0.1531, -0.7279,
            0.8348, -0.4336, -0.1253, -1.0069, 0.2815, -0.3406, 1.4044,
            -1.6412, 0.4354, -1.2269, 0.9194, 1.0373, 0.7552, 1.088,
        ]

        linear_regression = LinearRegression()
        linear_regression.fit(df, y_fd)

        np.testing.assert_allclose(
            linear_regression.basis_coefs[0].ravel(), beta_const_R, atol=0.001,
        )
        np.testing.assert_allclose(
            linear_regression.basis_coefs[1].ravel(), beta_atlantic_R, atol=0.001,
        )
        np.testing.assert_allclose(
            linear_regression.basis_coefs[2].ravel(), beta_continental_R, atol=0.001,
        )
        np.testing.assert_allclose(
            linear_regression.basis_coefs[3].ravel(), beta_pacific_R, atol=0.001,
        )
        np.testing.assert_equal(linear_regression.coef_[0].basis, y_fd.basis)

    def test_functional_covariates_concurrent(self) -> None:  # noqa: N802
        """
        Test a example of concurrent functional regression.

        Functional response with functional and multivariate covariates.
        Concurrent model.
        """
        y_basis = MonomialBasis(n_basis=2)
        x_basis = MonomialBasis(n_basis=3)

        X_train = pd.DataFrame({
            "covariate_1": FDataBasis(
                basis=x_basis,
                coefficients=[
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
            ),
            "covariate_2": [3, 1, 4],
            "covariate_3": [9, 2, 7],
        })
        y_train = FDataBasis(
            basis=y_basis,
            coefficients=[
                [1, 1],
                [2, 1],
                [3, 1],
            ],
        )

        expected_coefs = [
            FDataBasis(
                basis=y_basis,
                coefficients=[[5.608253, -2.866976]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[1.842478, -0.507984]],
            ),
            FDataBasis(
                basis=y_basis,
                coefficients=[[-0.55036, -0.032797]],
            ),
        ]

        X_test = pd.DataFrame({
            "covariate_1": FDataBasis(
                basis=x_basis,
                coefficients=[
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 1, 0],
                ],
            ),
            "covariate_2": [2, 1, 1],
            "covariate_3": [0, 2, 1],
        })
        y_test = FDataBasis(
            basis=y_basis,
            coefficients=[
                [3.323643, 2.012006],
                [0.380445, 2.454396],
                [7.378201, -0.666481],
            ],
        )

        _test_linear_regression_common(
            X_train=X_train,
            y_train=y_train,
            expected_coefs=expected_coefs,
            X_test=X_test,
            y_test=y_test,
            coef_basis=[y_basis, y_basis, y_basis],
        )

    def test_error_y_X_samples_different(self) -> None:  # noqa: N802
        """Number of response samples and explanatory samples are not different.

        Raises ValueError when response is functional.
        """
        y_basis = MonomialBasis(n_basis=2)
        X = [[1, 2], [3, 4], [5, 6], [1, 0]]

        y_fd = FDataBasis(y_basis, [[1, 2], [3, 4], [5, 6]])

        funct_reg = LinearRegression()
        with np.testing.assert_raises(ValueError):
            funct_reg.fit(X, y_fd)


class TestHistoricalLinearRegression(unittest.TestCase):
    """Tests for historical linear regression."""

    def setUp(self) -> None:
        """Generate data according to the model."""
        self.random = np.random.RandomState(1)

        self.n_samples = 50
        self.n_features = 20
        self.intercept = make_gaussian_process(
            n_samples=1,
            n_features=self.n_features,
            cov=Gaussian(length_scale=0.4),
            random_state=self.random,
        )

        self.X = make_gaussian_process(
            n_samples=self.n_samples,
            n_features=self.n_features,
            cov=Gaussian(length_scale=0.4),
            random_state=self.random,
        )

        self.coefficients = make_gaussian(
            n_samples=1,
            grid_points=[np.linspace(0, 1, self.n_features)] * 2,
            cov=Gaussian(length_scale=1),
            random_state=self.random,
        )

        self.X2 = make_gaussian_process(
            n_samples=self.n_samples,
            n_features=self.n_features,
            cov=Gaussian(length_scale=0.4),
            random_state=self.random,
        )

        self.coefficients2 = make_gaussian(
            n_samples=1,
            grid_points=[np.linspace(0, 1, self.n_features)] * 2,
            cov=Gaussian(length_scale=1),
            random_state=self.random,
        )

        self.create_model()
        self.create_vectorial_model()

    def create_model_no_intercept(
        self,
        X: FDataGrid,
        coefficients: FDataGrid,
    ) -> FDataGrid:
        """Create a functional response according to historical model."""
        integral_body = (
            X.data_matrix[..., 0, np.newaxis]
            * coefficients.data_matrix[..., 0]
        )
        integral_matrix = cumulative_trapezoid(
            integral_body,
            x=X.grid_points[0],
            initial=0,
            axis=1,
        )
        integral = np.diagonal(integral_matrix, axis1=1, axis2=2)
        return X.copy(data_matrix=integral)

    def create_model(self) -> None:
        """Create a functional response according to historical model."""
        model_no_intercept = self.create_model_no_intercept(
            X=self.X,
            coefficients=self.coefficients,
        )
        self.y = model_no_intercept + self.intercept

    def create_vectorial_model(self) -> None:
        """Create a functional response according to historical model."""
        model_no_intercept = self.create_model_no_intercept(
            X=self.X,
            coefficients=self.coefficients,
        )
        model_no_intercept2 = self.create_model_no_intercept(
            X=self.X2,
            coefficients=self.coefficients2,
        )
        self.y2 = model_no_intercept + model_no_intercept2 + self.intercept

    def test_historical(self) -> None:
        """Test historical regression with data following the model."""
        regression = HistoricalLinearRegression(n_intervals=6)
        fit_predict_result = regression.fit_predict(self.X, self.y)
        predict_result = regression.predict(self.X)

        np.testing.assert_allclose(
            predict_result.data_matrix,
            fit_predict_result.data_matrix,
        )

        np.testing.assert_allclose(
            predict_result.data_matrix,
            self.y.data_matrix,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            regression.intercept_.data_matrix,
            self.intercept.data_matrix,
            rtol=1e-3,
        )

        np.testing.assert_allclose(
            regression.coef_.data_matrix[0, ..., 0],
            np.triu(self.coefficients.data_matrix[0, ..., 0]),
            atol=0.35,
            rtol=0,
        )

    def test_historical_vectorial(self) -> None:
        """Test historical regression with data following the vector model."""
        X = self.X.concatenate(self.X2, as_coordinates=True)

        regression = HistoricalLinearRegression(n_intervals=10)
        fit_predict_result = regression.fit_predict(X, self.y2)
        predict_result = regression.predict(X)

        np.testing.assert_allclose(
            predict_result.data_matrix,
            fit_predict_result.data_matrix,
        )

        np.testing.assert_allclose(
            predict_result.data_matrix,
            self.y2.data_matrix,
            atol=1e-1,
            rtol=0,
        )

        np.testing.assert_allclose(
            regression.intercept_.data_matrix,
            self.intercept.data_matrix,
            rtol=1e-2,
        )

        # Coefficient matrix not tested as it is probably
        # an ill-posed problem


if __name__ == '__main__':
    unittest.main()
