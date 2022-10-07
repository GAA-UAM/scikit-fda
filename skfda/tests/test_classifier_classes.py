"""Tests classes attribute of classifiers."""

import unittest
from typing import List

import numpy as np

from skfda._utils._sklearn_adapter import ClassifierMixin
from skfda.datasets import make_multimodal_samples
from skfda.representation import FData

from ..typing._numpy import NDArrayAny


class TestClassifierClasses(unittest.TestCase):
    """Test for classifiers classes."""

    def setUp(self) -> None:
        """Establish train and test data sets."""
        # List of classes to test
        # Adding new classes to this list will test the classifiers with them
        self.classes_list: List[NDArrayAny] = [
            np.array([0, 1, 2]),
            np.array(["class_a", "class_b", "class_c"]),
        ]

        # Create one target y data of length n_samples from each class set
        n_samples = 30
        self.y_list = [
            np.resize(classes, n_samples)
            for classes in self.classes_list
        ]
        self.X = make_multimodal_samples(
            n_samples=n_samples,
        )

        self.tested_classifiers: List[ClassifierMixin[FData, NDArrayAny]] = [
        ]

    def test_classes(self) -> None:
        """Check classes attribute."""
        # Iterate over all classifiers with index
        for clf in self.tested_classifiers:
            # Iterate over all class sets to test different types of classes
            for classes, y in zip(self.classes_list, self.y_list):
                message = (
                    f'Testing classifier {clf.__class__.__name__} '
                    f'with classes {classes}'
                )
                with self.subTest(msg=message):
                    clf.fit(self.X, y)
                    np.testing.assert_array_equal(clf.classes_, classes)


if __name__ == "__main__":
    unittest.main()
