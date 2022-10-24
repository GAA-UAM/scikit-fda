"""Tests classes attribute of classifiers."""

import unittest
from typing import List

import numpy as np

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda.datasets import make_gaussian_process
from skfda.ml.classification import LogisticRegression
from skfda.representation import FData

from ..typing._numpy import NDArrayAny


class TestClassifierClasses(unittest.TestCase):
    """Test for classifiers classes."""

    def setUp(self) -> None:
        """Establish train and test data sets."""
        # List of classes to test
        # Adding new classes to this list will test the classifiers with them
        self.classes_list: List[NDArrayAny] = [
            np.array([0, 1]),
            np.array(["class_a", "class_b"]),
        ]

        # Create one target y data of length n_samples from each class set
        n_samples = 30
        self.y_list = [
            np.resize(classes, n_samples)
            for classes in self.classes_list
        ]
        self.X = make_gaussian_process(
            n_samples=n_samples,
            noise=0.05,
        )

        self.tested_classifiers: List[ClassifierMixin[FData, NDArrayAny]] = [
            LogisticRegression(),
        ]

    def test_classes(self) -> None:
        """Check classes attribute."""
        # Iterate over all classifiers with index
        for clf in self.tested_classifiers:
            # Iterate over all class sets to test different types of classes
            for classes, y in zip(self.classes_list, self.y_list):
                with self.subTest(classifier=clf, classes=classes):
                    clf.fit(self.X, y)
                    np.testing.assert_array_equal(clf.classes_, classes)


if __name__ == "__main__":
    unittest.main()
