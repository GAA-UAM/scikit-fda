"""Test to check the per class feature transformer module."""

import unittest

import numpy as np

from skfda._utils import _classifier_get_classes
from skfda.datasets import fetch_growth
from skfda.ml.classification import KNeighborsClassifier
from skfda.preprocessing.dim_reduction.feature_extraction import (
    PerClassTransformer,
)
from skfda.preprocessing.dim_reduction.variable_selection import (
    RecursiveMaximaHunting,
)


class TestPCT(unittest.TestCase):
    def setUp(self) -> None:
        X, y = fetch_growth(return_X_y=True, as_frame=True)
        self.X = X.iloc[:, 0].values
        self.y = y.values.codes

    def test_transform(self) -> None:

        t = PerClassTransformer(
            RecursiveMaximaHunting(),
            array_output=True,
        )
        t.fit_transform(self.X, self.y)
        transformed = t.transform(self.X)

        classes, y_ind = _classifier_get_classes(self.y)
        for cur_class in range(classes.size):
            feature_transformer = RecursiveMaximaHunting().fit(
                self.X[y_ind == cur_class],
                self.y[y_ind == cur_class],
            )
            a = feature_transformer.transform(self.X)
            np.testing.assert_array_equal(transformed[cur_class], a)

    def test_not_transformer_argument(self) -> None:

        t = PerClassTransformer(KNeighborsClassifier())
        self.assertRaises(
            TypeError,
            t.fit,
            self.X,
            self.y,
        )


if __name__ == '__main__':
    unittest.main()
