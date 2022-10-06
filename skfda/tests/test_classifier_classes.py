"""Tests classes attribute of classifiers."""

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda._utils._utils import _classifier_get_classes
from skfda.datasets import fetch_growth
from skfda.exploratory.stats.covariance import ParametricGaussianCovariance
from skfda.misc.covariances import Gaussian
from skfda.misc.metrics import l2_distance
from skfda.ml.classification import (
    DDClassifier,
    DDGClassifier,
    DTMClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    MaximumDepthClassifier,
    NearestCentroid,
    QuadraticDiscriminantAnalysis,
    RadiusNeighborsClassifier,
)
from skfda.ml.classification._depth_classifiers import _ArgMaxClassifier
from skfda.representation import FData

from ..typing._numpy import NDArrayAny


class TestClassifierClasses(unittest.TestCase):
    """Test for classifiers classes."""

    def setUp(self) -> None:
        """Establish train and test data sets."""
        X, y = fetch_growth(return_X_y=True)
        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=0,
        )
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train

        self.classes = _classifier_get_classes(self._y_train)[0]
        self.tested_classifiers: list[ClassifierMixin[FData, NDArrayAny]] = [
            KNeighborsClassifier(),
            RadiusNeighborsClassifier(),
            NearestCentroid(),
            DDGClassifier(multivariate_classifier=KNeighborsClassifier()),
            DDClassifier(degree=2),
            DTMClassifier(proportiontocut=0, metric=l2_distance),
            MaximumDepthClassifier(),
            _ArgMaxClassifier(),
            LogisticRegression(max_iter=1000),
            QuadraticDiscriminantAnalysis(
                cov_estimator=ParametricGaussianCovariance(
                    Gaussian(variance=6, length_scale=1),
                ),
            ),
        ]

    def test_classes(self) -> None:
        """Check classes attribute."""
        # Iterate over all classifiers with index
        for i, clf in enumerate(self.tested_classifiers):
            with self.subTest(i=i):
                clf.fit(self._X_train, self._y_train)
                np.testing.assert_array_equal(clf.classes_, self.classes)


if __name__ == "__main__":
    unittest.main()
