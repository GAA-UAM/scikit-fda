import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from skfda.datasets import fetch_growth
from skfda.ml.classification import DTMClassifier


class TestClassification(unittest.TestCase):

    def setUp(self) -> None:
        X, y = fetch_growth(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0)

    def test_dtm_independent_copy(self) -> None:

        clf = DTMClassifier(proportiontocut=0.25)
        clf1 = clone(clf)
        clf2 = DTMClassifier(proportiontocut=0.75)
        clf1.proportiontocut = 0.75
        clf1.fit(self.X_train, self.y_train)
        clf2.fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(
            clf1.predict(self.X_test), clf2.predict(self.X_test)
        )


if __name__ == '__main__':
    print()
    unittest.main()
