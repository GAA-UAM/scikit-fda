"""Test for scoring module."""
import math
import unittest
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics

from skfda import FDataBasis, FDataGrid
from skfda.misc.scoring import (
    ScoreFunction,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
from skfda.typing._numpy import NDArrayFloat

score_functions: Sequence[ScoreFunction] = (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


def _create_data_basis() -> Tuple[FDataBasis, FDataBasis]:
    coef_true = [[1, 2, 3], [4, 5, 6]]
    coef_pred = [[1, 2, 3], [4, 6, 5]]

    # y_true: 1) 1 + 2x + 3x^2
    #         2) 4 + 5x + 6x^2
    y_true = FDataBasis(
        basis=MonomialBasis(domain_range=((0, 3),), n_basis=3),
        coefficients=coef_true,
    )

    # y_pred: 1) 1 + 2x + 3x^2
    #         2) 4 + 6x + 5x^2
    y_pred = FDataBasis(
        basis=MonomialBasis(domain_range=((0, 3),), n_basis=3),
        coefficients=coef_pred,
    )

    # y_true - y_pred: 1) 0
    #                  2) -x + x^2
    return y_true, y_pred


def _create_data_grid() -> Tuple[FDataGrid, FDataGrid]:

    y_true, y_pred = _create_data_basis()
    grid = np.linspace(*y_true.domain_range[0], 100)

    return y_true.to_grid(grid), y_pred.to_grid(grid)


class TestScoreFunctionsGrid(unittest.TestCase):
    """Tests for score functions with FDataGrid representation."""

    def _test_generic_grid(
        self,
        function: ScoreFunction,
        weight: Optional[NDArrayFloat] = None,
        **kwargs: Any,
    ) -> None:
        y_true, y_pred = _create_data_grid()

        score = function(
            y_true,
            y_pred,
            multioutput='raw_values',
            sample_weight=weight,
            **kwargs,
        )

        score_sklearn = function(
            y_true.data_matrix.squeeze(),
            y_pred.data_matrix.squeeze(),
            multioutput='raw_values',
            sample_weight=weight,
            **kwargs,
        )

        np.testing.assert_allclose(
            score.data_matrix.squeeze(),
            score_sklearn,
        )

    def test_all(self) -> None:
        """Test all score functions."""
        for score_function in score_functions:
            with self.subTest(function=score_function):

                self._test_generic_grid(score_function)

                try:
                    self._test_generic_grid(
                        score_function,
                        squared=False,
                    )
                except TypeError:
                    pass

                self._test_generic_grid(
                    score_function,
                    weight=np.array([3, 1]),
                )


class TestScoreFunctionGridBasis(unittest.TestCase):
    """Compare the results obtained for FDataGrid and FDataBasis."""

    def _test_grid_basis_generic(
        self,
        score_function: ScoreFunction,
        weight: Optional[NDArrayFloat] = None,
        **kwargs: Any,
    ) -> None:
        y_true_grid, y_pred_grid = _create_data_grid()

        y_true_basis = y_true_grid.to_basis(basis=BSplineBasis(n_basis=10))
        y_pred_basis = y_pred_grid.to_basis(basis=BSplineBasis(n_basis=10))

        # The results should be close but not equal as the two representations
        # do not give same functions.
        precision = 2

        score_grid = score_function(
            y_true_grid,
            y_pred_grid,
            sample_weight=weight,
            **kwargs,
        )
        score_basis = score_function(
            y_true_basis,
            y_pred_basis,
            sample_weight=weight,
            **kwargs,
        )

        self.assertAlmostEqual(score_basis, score_grid, places=precision)

    def test_all(self) -> None:
        """Test all score functions."""
        for score_function in score_functions:
            with self.subTest(function=score_function):

                self._test_grid_basis_generic(score_function)

                try:
                    self._test_grid_basis_generic(
                        score_function,
                        squared=False,
                    )
                except TypeError:
                    pass

                self._test_grid_basis_generic(
                    score_function,
                    weight=np.array([3, 1]),
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
        self.assertAlmostEqual(ev, 0.992968, places=6)

    def test_mean_absolut_error_basis(self) -> None:
        """Test Mean Absolute Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mae = mean_absolute_error(y_true, y_pred)

        # integrate 1/2 * | -x + x^2|
        self.assertAlmostEqual(mae, 0.8055555555)

    def test_mean_absolute_percentage_error_basis(self) -> None:
        """Test Mean Absolute Percentage Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # integrate |1/2 * (-x  + x^2) / (4 + 5x + 6x^2)|
        self.assertAlmostEqual(mape, 0.0199192187)

    def test_mean_squared_error_basis(self) -> None:
        """Test Mean Squared Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        mse = mean_squared_error(y_true, y_pred)

        # integrate 1/2 * (-x + x^2)^2
        self.assertAlmostEqual(mse, 2.85)

    def test_mean_squared_log_error_basis(self) -> None:
        """Test Mean Squared Log Error for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        msle = mean_squared_log_error(y_true, y_pred)

        # integrate 1/2*(log(1 + 4 + 5x + 6x^2) - log(1 + 4 + 6x + 5x^2))^2
        self.assertAlmostEqual(msle, 0.00107583)

    def test_r2_score_basis(self) -> None:
        """Test R2 Score for FDataBasis."""
        y_true, y_pred = _create_data_basis()

        r2 = r2_score(y_true, y_pred)

        # integrate 1 - num/den
        # where     num = 1/2*(-x + x^2)^2,
        # and       den = (1.5 + 1.5x + 1.5x^2)^2
        self.assertAlmostEqual(r2, 0.9859362)


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
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        grid_points = np.linspace(0, 2, 9)

        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        # 0/0 for FDataGrid
        np.testing.assert_almost_equal(
            r2_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).data_matrix.squeeze(),
            sklearn.metrics.r2_score(
                y_true_grid.data_matrix.squeeze(),
                y_pred_grid.data_matrix.squeeze(),
                multioutput='raw_values',
            ),
        )

        # 0/0 for FDataBasis
        self.assertAlmostEqual(
            r2_score(y_true_basis, y_pred_basis),
            -16.5,
        )

        # Case when numerator is non-zero and denominator is zero (in t = 1)
        # for all f in y_true, f(1) = 1
        # for all f in y_pred, f(1) = 2
        basis_coef_pred = [[2, 0], [3, -1], [4, -2]]
        y_pred_basis = FDataBasis(
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        # r/0 for FDataGrid (r != 0)
        np.testing.assert_almost_equal(
            r2_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            )(1),
            [[[-math.inf]]],
        )

        # r/0 for FDataBasis (r != 0)
        self.assertAlmostEqual(
            r2_score(
                y_true_basis,
                y_pred_basis,
            ),
            -math.inf,
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
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )

        grid_points = np.linspace(0, 2, 9)

        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        # 0/0 for FDataGrid
        np.testing.assert_almost_equal(
            explained_variance_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            ).data_matrix.flatten(),
            sklearn.metrics.explained_variance_score(
                y_true_grid.data_matrix.squeeze(),
                y_pred_grid.data_matrix.squeeze(),
                multioutput='raw_values',
            ),
        )

        # 0/0 for FDataBasis
        self.assertAlmostEqual(
            explained_variance_score(y_true_basis, y_pred_basis),
            -3,
        )

        # Case when numerator is non-zero and denominator is zero (in t = 1)
        basis_coef_pred = [[2, 0], [2, -1], [2, -2]]
        y_pred_basis = FDataBasis(
            basis=MonomialBasis(domain_range=((0, 2),), n_basis=2),
            coefficients=basis_coef_pred,
        )
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        # r/0 for FDataGrid (r != 0)
        np.testing.assert_almost_equal(
            explained_variance_score(
                y_true_grid,
                y_pred_grid,
                multioutput='raw_values',
            )(1),
            [[[-math.inf]]],
        )

        # r/0 for FDataBasis (r != 0)
        self.assertAlmostEqual(
            explained_variance_score(
                y_true_basis,
                y_pred_basis,
            ),
            -math.inf,
        )

    def test_zero_mape(self) -> None:
        """Test Mean Absolute Percentage Error when y_true can be zero."""
        basis_coef_true = [[3, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
        basis_coef_pred = [[1, 0, 0, 0, 0], [0, 0, 1, 1, 0]]

        # Fourier basis defined in (0, 2) with 3 elements
        # The functions are
        # y_true(t) = 1) 3/sqrt(2)
        #             2) 1/sqrt(2) * cos(pi t)
        #
        # y_pred(t) = 1) 1/sqrt(2)
        #             2) 1/sqrt(2) * cos(pi t) + 1/sqrt(2) * sin(2 pi t)
        # The second function in y_true should be zero at t = 0.5 and t = 1.5

        y_true_basis = FDataBasis(
            basis=FourierBasis(domain_range=((0, 2),), n_basis=5),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=FourierBasis(domain_range=((0, 2),), n_basis=5),
            coefficients=basis_coef_pred,
        )

        with self.assertWarns(RuntimeWarning):
            mean_absolute_percentage_error(y_true_basis, y_pred_basis)

        grid_points = np.linspace(0, 2, 9)

        # The set of points in which the functions are evaluated
        # includes 0.5 and 1.5
        y_true_grid = y_true_basis.to_grid(grid_points=grid_points)
        y_pred_grid = y_pred_basis.to_grid(grid_points=grid_points)

        with self.assertWarns(RuntimeWarning):
            mean_absolute_percentage_error(y_true_grid, y_pred_grid)

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
            basis=FourierBasis(domain_range=((0, 2),), n_basis=3),
            coefficients=basis_coef_true,
        )

        y_pred_basis = FDataBasis(
            basis=FourierBasis(domain_range=((0, 2),), n_basis=3),
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
