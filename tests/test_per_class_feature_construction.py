"""Test to check the per class feature transformer module"""
from skfda.datasets import fetch_growth
from skfda.ml.classification import KNeighborsClassifier
from skfda.preprocessing.dim_reduction.feature_extraction.\
    _per_class_feature_transformer import PerClassFeatureTransformer
from skfda.preprocessing.dim_reduction.variable_selection \
    import RecursiveMaximaHunting
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda._utils import _classifier_get_classes

import numpy as np
import pytest
import unittest


class TestPCFT(unittest.TestCase):

    # This test fails because the transformers do not have yet tags implemented
    @pytest.mark.skip(reason="Tags are not yet implemented on transformers")
    def test_transform(self):

        X, y = fetch_growth(return_X_y=True, as_frame=True)
        X = X.iloc[:, 0].values
        y = y.values.codes
        t = PerClassFeatureTransformer(RecursiveMaximaHunting())
        t.fit_transform(X, y)
        transformed = t.transform(X)

        classes, y_ind = _classifier_get_classes(y)
        for cur_class in range(classes.size):
            feature_transformer = RecursiveMaximaHunting().fit(
                X[y_ind == cur_class],
                y[y_ind == cur_class],
            )
            a = feature_transformer.transform(X)
            np.testing.assert_array_equal(transformed[cur_class], a)

    def test_not_transformer_argument(self):
        self.assertRaises(
            TypeError,
            PerClassFeatureTransformer,
            KNeighborsClassifier(),
        )

    def test_not_taget_required_fitting(self):
        self.assertRaises(TypeError, PerClassFeatureTransformer, FPCA())


if __name__ == '__main__':
    unittest.main()
