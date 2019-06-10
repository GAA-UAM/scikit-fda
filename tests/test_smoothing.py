import unittest
from skfda._utils import _check_estimator
import skfda.preprocessing.smoothing.kernel_smoothers as kernel_smoothers


class TestSklearnEstimators(unittest.TestCase):

    def test_nadaraya_watson(self):
        _check_estimator(kernel_smoothers.NadarayaWatsonSmoother)

    def test_local_linear_regression(self):
        _check_estimator(kernel_smoothers.LocalLinearRegressionSmoother)

    def test_knn(self):
        _check_estimator(kernel_smoothers.KNeighborsSmoother)
