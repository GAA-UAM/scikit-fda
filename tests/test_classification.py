"""Tests of classification methods."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from skfda.datasets import fetch_growth
from skfda.ml.classification import DTMClassifier

from skfda.representation import FData


class TestCentroidClassifiers(unittest.TestCase):
    """Tests for centroid classifiers."""

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

        np.testing.assert_array_equal(  # type: ignore
            clf1.predict(self._X_test),
            clf2.predict(self._X_test),
        )


if __name__ == '__main__':
    unittest.main()
