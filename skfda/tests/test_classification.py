"""Tests of classification methods."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier

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


class TestClassifiers(unittest.TestCase):
    """Tests for classifiers."""

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

    def test_dtm_independent_copy(self) -> None:
        """Check that copies are un-linked."""
        clf: DTMClassifier[FData] = DTMClassifier(proportiontocut=0.25)
        clf1 = clone(clf)
        clf2: DTMClassifier[FData] = DTMClassifier(proportiontocut=0.75)

        clf1.proportiontocut = 0.75
        clf1.fit(self._X_train, self._y_train)
        clf2.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf1.predict(self._X_test),
            clf2.predict(self._X_test),
        )

    def test_dtm_classifier(self) -> None:
        """Check DTM classifier."""
        clf: DTMClassifier[FData] = DTMClassifier(proportiontocut=0.25)
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
            ],
        )

    def test_centroid_classifier(self) -> None:
        """Check NearestCentroid classifier."""
        clf: NearestCentroid[FData] = NearestCentroid()
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
            ],
        )

    def test_dtm_inheritance(self) -> None:
        """Check that DTM is a subclass of NearestCentroid."""
        clf1: NearestCentroid[FData] = NearestCentroid()
        clf2: DTMClassifier[FData] = DTMClassifier(
            proportiontocut=0,
            metric=l2_distance,
        )
        clf1.fit(self._X_train, self._y_train)
        clf2.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf1.predict(self._X_test),
            clf2.predict(self._X_test),
        )

    def test_maximumdepth_classifier(self) -> None:
        """Check MaximumDepth classifier."""
        clf: MaximumDepthClassifier[FData] = MaximumDepthClassifier()
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
            ],
        )

    def test_dd_classifier(self) -> None:
        """Check DD classifier."""
        clf: DDClassifier[FData] = DDClassifier(degree=2)
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
            ],
        )

    def test_ddg_classifier(self) -> None:
        """Check DDG classifier."""
        clf: DDGClassifier[FData] = DDGClassifier(
            multivariate_classifier=_KNeighborsClassifier(),
        )
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
            ],
        )

    def test_maximumdepth_inheritance(self) -> None:
        """Check that MaximumDepth is a subclass of DDG."""
        clf1: DDGClassifier[FData] = DDGClassifier(
            multivariate_classifier=_ArgMaxClassifier(),
        )
        clf2: MaximumDepthClassifier[FData] = MaximumDepthClassifier()
        clf1.fit(self._X_train, self._y_train)
        clf2.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf1.predict(self._X_test),
            clf2.predict(self._X_test),
        )

    def test_kneighbors_classifier(self) -> None:
        """Check KNeighbors classifier."""
        clf = KNeighborsClassifier()
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            ],
        )

    def test_radiusneighbors_classifier(self) -> None:
        """Check RadiusNeighbors classifier."""
        clf = RadiusNeighborsClassifier(radius=15)
        clf.fit(self._X_train, self._y_train)

        np.testing.assert_array_equal(
            clf.predict(self._X_test),
            [  # noqa: WPS317
                0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
            ],
        )

    def test_radiusneighbors_small_raidus(self) -> None:
        """Check that an error is raised if radius too small."""
        clf = RadiusNeighborsClassifier(radius=1)
        clf.fit(self._X_train, self._y_train)

        with np.testing.assert_raises(ValueError):
            clf.predict(self._X_test)


if __name__ == '__main__':
    unittest.main()
