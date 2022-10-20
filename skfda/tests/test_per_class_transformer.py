"""Test to check the per class transformer module."""

import unittest

import numpy as np

from skfda._utils import _classifier_get_classes
from skfda.datasets import fetch_growth
from skfda.ml.classification import KNeighborsClassifier
from skfda.preprocessing.dim_reduction.variable_selection import (
    RecursiveMaximaHunting,
)
from skfda.preprocessing.feature_construction import PerClassTransformer
from skfda.representation import FDataGrid


class TestPerClassTransformer(unittest.TestCase):
    """Tests for PCT module."""

    def setUp(self) -> None:
        """Fetch the Berkeley Growth Study dataset."""
        X, y = fetch_growth(return_X_y=True, as_frame=True)
        self.X = X.iloc[:, 0].values
        self.y = y.values.codes

    def test_transform(self) -> None:
        """Check the data transformation is done correctly."""
        t = PerClassTransformer[FDataGrid, np.typing.NDArray[np.float_]](
            RecursiveMaximaHunting(),
            array_output=True,
        )
        t.fit_transform(self.X, self.y)
        transformed = t.transform(self.X)

        manual = np.empty((93, 0))
        classes, y_ind = _classifier_get_classes(self.y)
        for cur_class in range(classes.size):
            feature_transformer = RecursiveMaximaHunting().fit(
                self.X[y_ind == cur_class],
                self.y[y_ind == cur_class],
            )
            aux = np.array(feature_transformer.transform(self.X))
            manual = np.hstack((manual, aux))

        np.testing.assert_array_equal(transformed, manual)

    def test_not_transformer_argument(self) -> None:
        """Check that invalid arguments in fit raise exception."""
        t = PerClassTransformer[FDataGrid, np.typing.NDArray[np.float_]](
            KNeighborsClassifier(),
        )
        self.assertRaises(
            TypeError,
            t.fit,
            self.X,
            self.y,
        )


if __name__ == '__main__':
    unittest.main()
