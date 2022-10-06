"""Tests classes attribute of classifiers."""

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from skfda._utils._utils import _classifier_get_classes
from skfda.datasets import fetch_growth
from skfda.misc.metrics import l2_distance
from skfda.ml.classification import (
    DDClassifier,
    DDGClassifier,
    DTMClassifier,
    KNeighborsClassifier,
    MaximumDepthClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from skfda.ml.classification._depth_classifiers import _ArgMaxClassifier
from skfda.representation import FData


class TestClassifierClasses(unittest.TestCase):
    """Test for classifiers classes."""

    def setUp(self) -> None:
        """Establish train and test data sets."""
        X, y = fetch_growth(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=0,
        )
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self.classes = _classifier_get_classes(self._y_train)[0]

    def test_classes_kneighbors(self) -> None:
        """Check classes attribute of KNeighborsClassifier."""
        clf: KNeighborsClassifier[FData] = KNeighborsClassifier()
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_radiusneighbors(self) -> None:
        """Check classes attribute of RadiusNeighborsClassifier."""
        clf: RadiusNeighborsClassifier[FData] = RadiusNeighborsClassifier()
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_nearestcentroid(self) -> None:
        """Check classes attribute of NearestCentroid."""
        clf: NearestCentroid[FData] = NearestCentroid()
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_ddg(self) -> None:
        """Check classes attribute of DDGClassifier."""
        clf: DDGClassifier[FData] = DDGClassifier(
            multivariate_classifier=KNeighborsClassifier(),
        )
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_dd(self) -> None:
        """Check classes attribute of DDClassifier."""
        clf: DDClassifier[FData] = DDClassifier(degree=2)
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_dtm(self) -> None:
        """Check classes attribute of DTMClassifier."""
        clf: DTMClassifier[FData] = DTMClassifier(
            proportiontocut=0,
            metric=l2_distance,
        )
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_maximumdepth(self) -> None:
        """Check classes attribute of MaximumDepthClassifier."""
        clf: MaximumDepthClassifier[FData] = MaximumDepthClassifier()
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)

    def test_classes_argmax(self) -> None:
        """Check classes attribute of _ArgMaxClassifier."""
        clf: _ArgMaxClassifier = _ArgMaxClassifier()
        clf.fit(self._X_train, self._y_train)
        np.testing.assert_array_equal(clf.classes_, self.classes)


if __name__ == "__main__":
    unittest.main()
