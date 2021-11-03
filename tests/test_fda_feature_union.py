"""Test to check the Fda Feature Union module."""

import unittest

from pandas import DataFrame, concat

from skfda.datasets import fetch_growth
from skfda.misc.operators import SRSF
from skfda.preprocessing.dim_reduction.feature_extraction import (
    FPCA,
    FdaFeatureUnion,
)
from skfda.preprocessing.smoothing.kernel_smoothers import (
    NadarayaWatsonSmoother,
)
from skfda.representation import EvaluationTransformer


class TestFdaFeatureUnion(unittest.TestCase):

    def setUp(self) -> None:
        self.X = fetch_growth(return_X_y=True)[0]

    def test_incompatible_array_output(self) -> None:

        u = FdaFeatureUnion(
            [("EvaluationT", EvaluationTransformer(None)), ("fpca", FPCA())],
            np_array_output=False,
        )
        self.assertRaises(TypeError, u.fit_transform, self.X)

    def test_incompatible_fdatagrid_output(self) -> None:

        u = FdaFeatureUnion(
            [("EvaluationT", EvaluationTransformer(None)), ("srsf", SRSF())],
            np_array_output=True,
        )
        self.assertRaises(TypeError, u.fit_transform, self.X)

    def test_correct_transformation_concat(self) -> None:
        u = FdaFeatureUnion(
            [("srsf1", SRSF()), ("smooth", NadarayaWatsonSmoother())],
        )
        created_frame = u.fit_transform(self.X)

        t1 = SRSF().fit_transform(self.X)
        t2 = NadarayaWatsonSmoother().fit_transform(self.X)

        frames = [
            DataFrame({t1.dataset_name.lower(): t1}),
            DataFrame({t2.dataset_name.lower(): t2}),
        ]
        true_frame = concat(frames, axis=1)
        result = True
        self.assertEqual(result, true_frame.equals(created_frame))


if __name__ == '__main__':
    unittest.main()
