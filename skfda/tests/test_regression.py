from __future__ import annotations

import unittest
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz

from skfda.datasets import make_gaussian, make_gaussian_process
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


class TestScalarLinearRegression(unittest.TestCase):

    def test_regression_single_explanatory(self) -> None:

        x_basis = MonomialBasis(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = FourierBasis(n_basis=5)
        beta_fd = FDataBasis(beta_basis, [1, 1, 1, 1, 1])
        y = np.array([
            0.9999999999999993,
            0.162381381441085,
            0.08527083481359901,
            0.08519946930844623,
            0.09532291032042489,
            0.10550022969639987,
            0.11382675064746171,
        ])

        scalar = LinearRegression(coef_basis=[beta_basis])
        scalar.fit(x_fd, y)
        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            beta_fd.coefficients,
        )
        np.testing.assert_allclose(
            scalar.intercept_,
            0.0,
            atol=1e-6,
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
            scalar.intercept_,
            0.0,
        )

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

    def test_regression_multiple_explanatory(self) -> None:
        y = np.array([1, 2, 3, 4, 5, 6, 7])

        X = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))

        beta1 = BSplineBasis(domain_range=(0, 1), n_basis=5)

        scalar = LinearRegression(coef_basis=[beta1])

        scalar.fit(X, y)

        np.testing.assert_allclose(
            scalar.intercept_.round(4),
            np.array([32.65]),
            rtol=1e-3,
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
            ]]),
            rtol=1e-3,
        )

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_regression_mixed(self) -> None:

        multivariate = np.array([
            [0, 0], [2, 7], [1, 7], [3, 9],
            [4, 16], [2, 14], [3, 5],
        ])

        X: Sequence[
            np.typing.NDArray[np.float_] | FDataBasis,
        ] = [
            multivariate,
            FDataBasis(
                MonomialBasis(n_basis=3),
                [
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 0, 1], [1, 0, 0], [0, 1, 0],
                    [0, 0, 1],
                ],
            ),
        ]

        intercept = 2
        coefs_multivariate = np.array([3, 1])
        coefs_functions = FDataBasis(
            MonomialBasis(n_basis=3),
            [[3, 0, 0]],
        )
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression()
        scalar.fit(X, y)

        np.testing.assert_allclose(
            scalar.intercept_,
            intercept,
            atol=0.01,
        )

        np.testing.assert_allclose(
            scalar.coef_[0],
            coefs_multivariate,
            atol=0.01,
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

    def test_regression_df_multivariate(self) -> None:  # noqa: D102

        multivariate1 = [0, 2, 1, 3, 4, 2, 3]
        multivariate2 = [0, 7, 7, 9, 16, 14, 5]
        multivariate = [list(obs) for obs in zip(multivariate1, multivariate2)]

        x_basis = MonomialBasis(n_basis=3)
        x_fd = FDataBasis(x_basis, [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                    [1, 0, 1], [1, 0, 0], [0, 1, 0],
                                    [0, 0, 1]])

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

    def test_regression_mixed_regularization(self) -> None:

        multivariate = np.array([
            [0, 0], [2, 7], [1, 7], [3, 9],
            [4, 16], [2, 14], [3, 5],
        ])

        X: Sequence[
            np.typing.NDArray[np.float_] | FDataBasis,
        ] = [
            multivariate,
            FDataBasis(
                MonomialBasis(n_basis=3),
                [
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 0, 1], [1, 0, 0], [0, 1, 0],
                    [0, 0, 1],
                ]),
        ]

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
            scalar.intercept_,
            intercept,
            atol=0.01,
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
                5.349035, 16.456464, 13.361185, 23.930295,
                32.650965, 23.961766, 16.29029,
            ],
            atol=0.01,
        )

    def test_regression_regularization(self) -> None:

        x_basis = MonomialBasis(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = FourierBasis(n_basis=5)
        beta_fd = FDataBasis(
            beta_basis,
            [1.0403, 0, 0, 0, 0],
        )
        y = np.array([
            1.0000684777229512,
            0.1623672257830915,
            0.08521053851548224,
            0.08514200869281137,
            0.09529138749665378,
            0.10549625973303875,
            0.11384314859153018,
        ])

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
            scalar.intercept_,
            -0.15,
            atol=1e-4,
        )

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_pred_compare, atol=1e-4)

        x_basis = MonomialBasis(n_basis=3)
        x_fd = FDataBasis(
            x_basis,
            [
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
        assert isinstance(scalar.coef_[0], FDataBasis)
        np.testing.assert_allclose(
            scalar.coef_[0].coefficients,
            beta_fd.coefficients,
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
            scalar_reg.intercept_,
            0.998,
            atol=0.001,
        )

        y_pred = scalar_reg.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_reg, atol=0.001)

    def test_error_X_not_FData(self) -> None:
        """Tests that at least one variable is an FData object."""
        x_fd = np.identity(7)
        y = np.zeros(7)

        scalar = LinearRegression(coef_basis=[FourierBasis(n_basis=5)])

        with np.testing.assert_warns(UserWarning):
            scalar.fit([x_fd], y)

    def test_error_y_is_FData(self) -> None:
        """Tests that none of the explained variables is an FData object."""
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = list(FDataBasis(MonomialBasis(n_basis=7), np.identity(7)))

        scalar = LinearRegression(coef_basis=[FourierBasis(n_basis=5)])

        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)  # type: ignore[arg-type]

    def test_error_X_beta_len_distinct(self) -> None:
        """Test that the number of beta bases and explanatory variables
        are not different """
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(7)])
        beta = FourierBasis(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd, x_fd], y)

        scalar = LinearRegression(coef_basis=[beta, beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_y_X_samples_different(self) -> None:
        """Test that the number of response samples and explanatory samples
        are not different """

        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(8)])
        beta = FourierBasis(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

        x_fd = FDataBasis(MonomialBasis(n_basis=8), np.identity(8))
        y = np.array([1 for _ in range(7)])
        beta = FourierBasis(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_beta_not_basis(self) -> None:
        """Test that all beta are Basis objects. """
        x_fd = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))
        y = np.array([1 for _ in range(7)])
        beta = FDataBasis(MonomialBasis(n_basis=7), np.identity(7))

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(TypeError):
            scalar.fit([x_fd], y)

    def test_error_weights_lenght(self) -> None:
        """Test that the number of weights is equal to n_samples."""
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
        integral_matrix = cumtrapz(
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
            atol=0.3,
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
