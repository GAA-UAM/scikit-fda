"""Test to check the Fda Feature Union module."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from skfda.datasets import fetch_growth
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.misc.operators import SRSF
from skfda.preprocessing.feature_construction import (
    EvaluationTransformer,
    FDAFeatureUnion,
)
from skfda.preprocessing.smoothing import KernelSmoother


class TestFDAFeatureUnion(unittest.TestCase):
    """Check the Fda Feature Union module."""

    def setUp(self) -> None:
        """Fetch the Berkeley Growth Study dataset."""
        self.X = fetch_growth(return_X_y=True)[0]

    def test_incompatible_fdatagrid_output(self) -> None:
        """Check that array_output fails for FData features."""
        u = FDAFeatureUnion(
            [("eval", EvaluationTransformer(None)), ("srsf", SRSF())],
            array_output=True,
        )

        with self.assertRaises(TypeError):
            u.fit_transform(self.X)

    def test_correct_transformation_concat(self) -> None:
        """Check that the transformation is done correctly."""
        u = FDAFeatureUnion(
            [
                ("srsf1", SRSF()),
                (
                    "smooth",
                    KernelSmoother(
                        kernel_estimator=NadarayaWatsonHatMatrix(),
                    ),
                ),
            ],
        )
        created_frame = u.fit_transform(self.X)

        t1 = SRSF().fit_transform(self.X)
        t2 = KernelSmoother(
            kernel_estimator=NadarayaWatsonHatMatrix(),
        ).fit_transform(self.X)

        true_frame = pd.concat(
            [
                pd.DataFrame({0: t1}),
                pd.DataFrame({0: t2}),
            ],
            axis=1,
        )
        assert_frame_equal(true_frame, created_frame)


if __name__ == '__main__':
    unittest.main()
