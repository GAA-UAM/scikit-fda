import unittest
from skfda._utils import _check_estimator
import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as kernel_smoothers
import skfda.preprocessing.smoothing.validation as validation

import numpy as np
import sklearn


class TestSklearnEstimators(unittest.TestCase):

    def test_nadaraya_watson(self):
        _check_estimator(kernel_smoothers.NadarayaWatsonSmoother)

    def test_local_linear_regression(self):
        _check_estimator(kernel_smoothers.LocalLinearRegressionSmoother)

    def test_knn(self):
        _check_estimator(kernel_smoothers.KNeighborsSmoother)


class _LinearSmootherLeaveOneOutScorerAlternative():
    r"""Alternative implementation of the LinearSmootherLeaveOneOutScorer"""

    def __call__(self, estimator, X, y):
        estimator_clone = sklearn.base.clone(estimator)

        estimator_clone._cv = True
        y_est = estimator_clone.fit_transform(X)

        return -np.mean((y.data_matrix[..., 0] - y_est.data_matrix[..., 0])**2)


class TestLeaveOneOut(unittest.TestCase):

    def _test_generic(self, estimator_class):
        loo_scorer = validation.LinearSmootherLeaveOneOutScorer()
        loo_scorer_alt = _LinearSmootherLeaveOneOutScorerAlternative()
        x = np.linspace(-2, 2, 5)
        fd = skfda.FDataGrid(x ** 2, x)

        estimator = estimator_class()

        grid = validation.SmoothingParameterSearch(
            estimator, [2, 3],
            scoring=loo_scorer)
        grid.fit(fd)
        score = np.array(grid.cv_results_['mean_test_score'])

        grid_alt = validation.SmoothingParameterSearch(
            estimator, [2, 3],
            scoring=loo_scorer_alt)
        grid_alt.fit(fd)
        score_alt = np.array(grid_alt.cv_results_['mean_test_score'])

        np.testing.assert_array_almost_equal(score, score_alt)

    def test_nadaraya_watson(self):
        self._test_generic(kernel_smoothers.NadarayaWatsonSmoother)

    def test_local_linear_regression(self):
        self._test_generic(kernel_smoothers.LocalLinearRegressionSmoother)

    def test_knn(self):
        self._test_generic(kernel_smoothers.KNeighborsSmoother)
