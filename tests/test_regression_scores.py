"""Test for Score Functions module."""
import unittest
from typing import Optional, Tuple

import numpy as np
import sklearn

from skfda import FDataBasis, FDataGrid
from skfda.datasets import fetch_tecator
from skfda.misc.score_functions import (
    ScoreFunction,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from skfda.representation._typing import NDArrayFloat
from skfda.representation.basis import BSpline, Fourier, Monomial


def _create_data_grid(n: int) -> Tuple[FDataGrid, FDataGrid]:
    X, y = fetch_tecator(return_X_y=True, as_frame=True)
    fd = X.iloc[:, 0].values

    y_true = fd[:n]
    y_pred = fd[n:2 * n]

    return y_true, y_pred


def _create_data_basis() -> Tuple[FDataBasis, FDataBasis]:
    coef_true = [[1, 2, 3], [4, 5, 6]]
    coef_pred = [[1, 2, 3], [4, 6, 5]]

    # y_true: 1) 1 + 2x + 3x^2
    #         2) 4 + 5x + 6x^2
    y_true = FDataBasis(
        basis=Monomial(domain_range=((0, 3),), n_basis=3),
        coefficients=coef_true,
    )

    # y_true: 1) 1 + 2x + 3x^2
    #         2) 4 + 6x + 5x^2
    y_pred = FDataBasis(
        basis=Monomial(domain_range=((0, 3),), n_basis=3),
        coefficients=coef_pred,
    )

    # y_true - y_pred: 1) 0
    #                  2) -x + x^2
    return y_true, y_pred


class TestScoreFunctionsGrid(unittest.TestCase):
    """Tests for score functions with FDataGrid representation."""

    n = 10

    def _test_generic_grid(
        self,
        function: ScoreFunction,
        sklearn_function: ScoreFunction,
        weight: Optional[NDArrayFloat] = None,
        squared: bool = True,
    ) -> None:
        y_true, y_pred = _create_data_grid(self.n)

        if squared:
            score = function(
                y_true,
                y_pred,
                multioutput='raw_values',
                sample_weight=weight,
            )

            score_sklearn = sklearn_function(
                y_true.data_matrix.reshape(
                    (y_true.data_matrix.shape[0], -1),
                ),
                y_pred.data_matrix.reshape(
                    (y_pred.data_matrix.shape[0], -1),
                ),
                multioutput='raw_values',
                sample_weight=weight,
            )
        else:
            score = function(
                y_true,
                y_pred,
                multioutput='raw_values',
                sample_weight=weight,
                squared=False,
            )

            score_sklearn = sklearn_function(
                y_true.data_matrix.reshape(
                    (y_true.data_matrix.shape[0], -1),
                ),
                y_pred.data_matrix.reshape(
                    (y_pred.data_matrix.shape[0], -1),
                ),
                multioutput='raw_values',
                sample_weight=weight,
                squared=False,
            )

        np.testing.assert_allclose(
            score.data_matrix.reshape(
                (score.data_matrix.shape[0], -1),
            )[0],
            score_sklearn,
        )

    def test_explained_variance_score_grid(self) -> None:
        """Test Explained Variance Score for FDataGrid."""
        self._test_generic_grid(
            explained_variance_score,
            sklearn.metrics.explained_variance_score,
        )

        self._test_generic_grid(
            explained_variance_score,
            sklearn.metrics.explained_variance_score,
            np.random.random_sample(self.n),
        )

    def test_mean_absolute_error_grid(self) -> None:
        """Test Mean Absolute Error for FDataGrid."""
        self._test_generic_grid(
            mean_absolute_error,
            sklearn.metrics.mean_absolute_error,
        )

        self._test_generic_grid(
            mean_absolute_error,
            sklearn.metrics.mean_absolute_error,
            np.random.random_sample(self.n),
        )

    def test_mean_absolute_percentage_error_grid(self) -> None:
        """Test Mean Absolute Percentage Error for FDataGrid."""
        self._test_generic_grid(
            mean_absolute_percentage_error,
            sklearn.metrics.mean_absolute_percentage_error,
        )

        self._test_generic_grid(
            mean_absolute_percentage_error,
            sklearn.metrics.mean_absolute_percentage_error,
            np.random.random_sample(self.n),
        )

    def test_mean_squared_error_grid(self) -> None:
        """Test Mean Squared Error for FDataGrid."""
        self._test_generic_grid(
            mean_squared_error,
            sklearn.metrics.mean_squared_error,
        )

        self._test_generic_grid(
            mean_squared_error,
            sklearn.metrics.mean_squared_error,
            squared=False,
        )

        self._test_generic_grid(
            mean_squared_error,
            sklearn.metrics.mean_squared_error,
            np.random.random_sample(self.n),
        )

    def test_mean_squared_log_error_grid(self) -> None:
        """Test Mean Squared Log Error for FDataGrid."""
        self._test_generic_grid(
            mean_squared_log_error,
            sklearn.metrics.mean_squared_log_error,
        )

        self._test_generic_grid(
            mean_squared_log_error,
            sklearn.metrics.mean_squared_log_error,
            squared=False,
        )

        self._test_generic_grid(
            mean_squared_log_error,
            sklearn.metrics.mean_squared_log_error,
            np.random.random_sample(self.n),
        )

    def test_r2_score_grid(self) -> None:
        """Test R2 Score for FDataGrid."""
        self._test_generic_grid(
            r2_score,
            sklearn.metrics.r2_score,
        )

        self._test_generic_grid(
            r2_score,
            sklearn.metrics.r2_score,
            np.random.random_sample(self.n),
        )


class TestScoreFunctionGridBasis(unittest.TestCase):
    """Compare the results obtained for FDataGrid and FDataBasis."""

    n = 10

    def _test_grid_basis_generic(
        self,
        score_function: ScoreFunction,
        sample_weight: Optional[NDArrayFloat] = None,
        squared: bool = True,
    ) -> None:
        y_true_grid, y_pred_grid = _create_data_grid(self.n)

        y_true_basis = y_true_grid.to_basis(basis=BSpline(n_basis=10))
        y_pred_basis = y_pred_grid.to_basis(basis=BSpline(n_basis=10))

        # The results should be close but not equal as the two representations
        # do not give same functions.
        precision = 2

        if squared:
            score_grid = score_function(
                y_true_grid,
                y_pred_grid,
                sample_weight=sample_weight,
            )
            score_basis = score_function(
                y_true_basis,
                y_pred_basis,
                sample_weight=sample_weight,
            )
        else:
            score_grid = score_function(
                y_true_grid,
                y_pred_grid,
                sample_weight=sample_weight,
                squared=False,
            )
            score_basis = score_function(
                y_true_basis,
                y_pred_basis,
                sample_weight=sample_weight,
                squared=False,
            )

        np.testing.assert_almost_equal(
            score_grid,
            score_basis,
            decimal=precision,
        )

    def test_explained_variance_score(self) -> None:
        """Explained variance score for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(explained_variance_score)
        self._test_grid_basis_generic(
            explained_variance_score,
            np.random.random_sample((self.n,)),
        )

    def test_mean_absolute_error(self) -> None:
        """Mean Absolute Error for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(mean_absolute_error)
        self._test_grid_basis_generic(
            mean_absolute_error,
            np.random.random_sample((self.n,)),
        )

    def test_mean_absolute_percentage_error(self) -> None:
        """Mean Absolute Percentage Error for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(mean_absolute_percentage_error)
        self._test_grid_basis_generic(
            mean_absolute_percentage_error,
            np.random.random_sample((self.n,)),
        )

    def test_mean_squared_error(self) -> None:
        """Mean Squared Error for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(mean_squared_error)
        self._test_grid_basis_generic(
            mean_squared_error,
            np.random.random_sample((self.n,)),
        )
        self._test_grid_basis_generic(mean_squared_error, squared=False)

    def test_mean_squared_log_error(self) -> None:
        """Mean Squared Log Error for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(mean_squared_log_error)
        self._test_grid_basis_generic(
            mean_squared_log_error,
            np.random.random_sample((self.n,)),
        )
        self._test_grid_basis_generic(mean_squared_log_error, squared=False)

    def test_r2_score(self) -> None:
        """R2 Score for FDataGrid and FDataBasis."""
        self._test_grid_basis_generic(r2_score)
        self._test_grid_basis_generic(
            r2_score,
            np.random.random_sample((self.n,)),
        )


class TestScoreFunctionsBasis(unittest.TestCase):
    """Tests for score functions with FDataBasis representation."""

    def test_explained_variance_basis(self) -> None:
        """Test Explain Variance Score for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        ev = explained_variance_score(y_true, y_pred)

        # integrate 1 - num/den
        # where     num = (1/2x -1/2x^2)^2
        # and       den = (1.5 + 1.5x + 1.5x^2)^2
        np.testing.assert_almost_equal(ev, 0.992968)

    def test_mean_absolut_error_basis(self) -> None:
        """Test Mean Absolute Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mae = mean_absolute_error(y_true, y_pred)

        # integrate 1/2 * | -x + x^2|
        np.testing.assert_almost_equal(mae, 0.8055555555)

    def test_mean_absolute_percentage_error_basis(self) -> None:
        """Test Mean Absolute Percentage Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # integrate |1/2 * (-x  + x^2) / (4 + 5x + 6x^2)|
        np.testing.assert_almost_equal(mape, 0.0199192187)

    def test_mean_squared_error_basis(self) -> None:
        """Test Mean Squared Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mse = mean_squared_error(y_true, y_pred)

        # integrate 1/2 * (-x + x^2)^2
        np.testing.assert_almost_equal(mse, 2.85)

    def test_mean_squared_log_error_basis(self) -> None:
        """Test Mean Squared Log Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        msle = mean_squared_log_error(y_true, y_pred)

        # integrate 1/2*(log(1 + 4 + 5x + 6x^2) - log(1 + 4 + 6x + 5x^2))^2
        np.testing.assert_almost_equal(msle, 0.00107583)

    def test_r2_score_basis(self) -> None:
        """Test R2 Score for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        r2 = r2_score(y_true, y_pred)

        # integrate 1 - num/den
        # where     num = 1/2*(-x + x^2)^2,
        # and       den = (1.5 + 1.5x + 1.5x^2)^2
        np.testing.assert_almost_equal(r2, 0.9859362)


class TestScoreZeroDenominator(unittest.TestCase):
    """Tests Score Functions with edge cases."""

    def test_zero_r2(self) -> None:
        """Test R2 Score when the denominator is zero."""
        # Case when both numerator and denominator is zero (in t = 1)
        basis_coef_true = [[0, 1], [-1, 2], [-2, 3]]
        basis_coef_pred = [[1, 0], [2, -1], [3, -2]]

        # y_true and y_pred are 2 sets of 3 straight lines
        # for all f, f(1) = 1
        y_true_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        grid_points = np.linspace(0, 2, 9)

        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        np.testing.assert_almost_equal(
            r2_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).evaluate(1),
            [[[0]]],
        )

        np.testing.assert_almost_equal(
            r2_score(y_true_basis, y_pred_basis),
            -16.5,
        )

        # Case when numerator is non-zero and denominator is zero (in t = 1)
        # for all f in y_true, f(1) = 1
        # for all f in y_pred, f(1) = 2
        basis_coef_pred = [[2, 0], [3, -1], [4, -2]]
        y_pred_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)
        np.testing.assert_almost_equal(
            r2_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).evaluate(1),
            [[[float('-inf')]]],
        )

    def test_zero_ev(self) -> None:
        """Test R2 Score when the denominator is zero."""
        basis_coef_true = [[0, 1], [-1, 2], [-2, 3]]
        basis_coef_pred = [[1, 0], [2, -1], [3, -2]]
        # Case when both numerator and denominator is zero (in t = 1)

        # y_true and y_pred are 2 sets of 3 straight lines
        # for all f, f(1) = 1
        # var(y_true(1)) = 0 and var(y_true(1) - y_pred(1)) = 0
        y_true_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        grid_points = np.linspace(0, 2, 9)

        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        np.testing.assert_almost_equal(
            explained_variance_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).evaluate(1),
            [[[0]]],
        )

        # Case when numerator is non-zero and denominator is zero (in t = 1)
        basis_coef_pred = [[2, 0], [2, -1], [2, -2]]
        y_pred_basis = FDataBasis(
            basis=Monomial(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)
        np.testing.assert_almost_equal(
            explained_variance_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).evaluate(1),
            [[[float('-inf')]]],
        )

    def test_zero_mape(self) -> None:
        """Test Mean Absolute Percentage Error when y_true can be zero."""
        basis_coef_true = [[3, 0, 0], [0, 0, 1]]
        basis_coef_pred = [[1, 0, 0], [1, 0, 1]]

        # Fourier basis defined in (0, 2) with 3 elements
        # The functions are
        # y_true(t) = 1) 3/sqrt(2)
        #             2) 1/sqrt(2) * cos(pi t)
        #
        # y_pred(t) = 1) 1/sqrt(2)
        #             2) 1/sqrt(2) + 1/sqrt(2) * cos(pi t)
        # The second function in y_true should be zero at t = 0.5 and t = 1.5

        y_true_basis = FDataBasis(
            basis=Fourier(domain_range=((0, 2),), n_basis=3),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=Fourier(domain_range=((0, 2),), n_basis=3),
            coefficients=basis_coef_pred,
        )

        self.assertWarns(
            RuntimeWarning,
            mean_absolute_percentage_error,
            y_true_basis,
            y_pred_basis,
        )

        grid_points = np.linspace(0, 2, 9)

        # The set of points in which the functions are evaluated
        # includes 0.5 and 1.5
        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        self.assertWarns(
            RuntimeWarning,
            mean_absolute_percentage_error,
            y_true_grid,
            y_pred_grid,
        )

    def test_negative_msle(self) -> None:
        """Test Mean Squared Log Error when there are negative data."""
        basis_coef_true = [[3, 0, 0], [0, 0, 1]]
        basis_coef_pred = [[1, 0, 0], [np.sqrt(2), 0, 1]]

        # Fourier basis defined in (0, 2) with 3 elements
        # The functions are
        # y_true(t) = 1) 3/sqrt(2)
        #             2) 1/sqrt(2) * cos(pi t)
        #
        # y_pred(t) = 1) 1/sqrt(2)
        #             2) 1 + 1/sqrt(2) * cos(pi t)
        # The second function in y_true should be negative
        # between t = 0.5 and t = 1.5
        # All functions in y_pred should be always positive

        y_true_basis = FDataBasis(
            basis=Fourier(domain_range=((0, 2),), n_basis=3),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=Fourier(domain_range=((0, 2),), n_basis=3),
            coefficients=basis_coef_pred,
        )

        self.assertRaises(
            ValueError,
            mean_squared_log_error,
            y_true_basis,
            y_pred_basis,
        )

        grid_points = np.linspace(0, 2, 9)

        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        self.assertRaises(
            ValueError,
            mean_squared_log_error,
            y_true_grid,
            y_pred_grid,
        )
