"""Tests for Recursive Maxima Hunting (RMH)."""
import unittest

import numpy as np
import pytest

import skfda
from skfda.datasets import make_gaussian_process
from skfda.exploratory.stats.covariance import ParametricGaussianCovariance
from skfda.misc.covariances import Exponential
from skfda.preprocessing.dim_reduction import variable_selection as vs


class TestRMH(unittest.TestCase):
    """Tests for RMH."""

    def test_gaussian_homoscedastic(self) -> None:
        """
        Test the case for which RMH is optimal.

        It tests a case with two homoscedastic Brownian processes where the
        difference of means is piecewise linear.

        In this case RMH should return the points where the linear parts
        join.

        """
        n_samples = 1000
        n_features = 101

        def mean_1(  # noqa: WPS430
            t: np.typing.NDArray[np.float64],
        ) -> np.typing.NDArray[np.float64]:

            return (  # type: ignore[no-any-return]
                np.abs(t - 0.25)
                - 2 * np.abs(t - 0.5)
                + np.abs(t - 0.75)
            )

        X_0 = make_gaussian_process(
            n_samples=n_samples // 2,
            n_features=n_features,
            random_state=0,
        )
        X_1 = make_gaussian_process(
            n_samples=n_samples // 2,
            n_features=n_features,
            mean=mean_1,
            random_state=1,
        )
        X = skfda.concatenate((X_0, X_1))

        y = np.zeros(n_samples)
        y[n_samples // 2:] = 1

        correction = vs.recursive_maxima_hunting.GaussianSampleCorrection()
        stopping_condition = vs.recursive_maxima_hunting.ScoreThresholdStop(
            threshold=0.05,
        )

        rmh = vs.RecursiveMaximaHunting(
            correction=correction,
            stopping_condition=stopping_condition,
        )
        rmh.fit(X, y)
        point_mask = rmh.get_support()
        points = X.grid_points[0][point_mask]
        np.testing.assert_allclose(points, [0.25, 0.5, 0.75], rtol=1e-2)

    @pytest.mark.filterwarnings(
        'ignore::sklearn.exceptions.ConvergenceWarning'
    )
    def test_fit_exponential(self) -> None:
        """
        Test the case for which RMH is optimal.

        It tests a case with two homoscedastic Brownian processes where the
        difference of means is piecewise linear.

        In this case RMH should return the points where the linear parts
        join.

        """
        n_samples = 1000
        n_features = 101

        def mean_1(  # noqa: WPS430
            t: np.typing.NDArray[np.float64],
        ) -> np.typing.NDArray[np.float64]:

            return (  # type: ignore[no-any-return]
                np.abs(t - 0.25)
                - 2 * np.abs(t - 0.5)
                + np.abs(t - 0.75)
            )

        X_0 = make_gaussian_process(
            n_samples=n_samples // 2,
            n_features=n_features,
            cov=Exponential(length_scale=2),
            random_state=0,
        )
        X_1 = make_gaussian_process(
            n_samples=n_samples // 2,
            n_features=n_features,
            mean=mean_1,
            cov=Exponential(length_scale=2),
            random_state=1,
        )
        X = skfda.concatenate((X_0, X_1))

        y = np.zeros(n_samples)
        y[n_samples // 2:] = 1

        correction = vs.recursive_maxima_hunting.GaussianSampleCorrection(
            cov_estimator=ParametricGaussianCovariance(
                cov=Exponential(),
            ),
        )
        stopping_condition = vs.recursive_maxima_hunting.ScoreThresholdStop(
            threshold=0.05,
        )

        rmh = vs.RecursiveMaximaHunting(
            correction=correction,
            stopping_condition=stopping_condition,
        )
        rmh.fit(X, y)
        point_mask = rmh.get_support()
        points = X.grid_points[0][point_mask]
        np.testing.assert_allclose(points, [0.25, 0.5, 0.75], rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
