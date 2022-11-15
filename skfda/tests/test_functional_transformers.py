"""Test to check the per functional transformers."""

import unittest

import numpy as np

import skfda.representation.basis as basis
from skfda.datasets import fetch_growth
from skfda.preprocessing.feature_construction import (
    unconditional_expected_value,
)


class TestUnconditionalExpectedValues(unittest.TestCase):
    """Tests for unconditional expected values method."""

    def test_transform(self) -> None:
        """Compare results with grid and basis representations."""
        X = fetch_growth(return_X_y=True)[0]

        data_grid = unconditional_expected_value(
            X[:5],
            np.log,
        )
        data_basis = unconditional_expected_value(
            X[:5].to_basis(basis.BSplineBasis(n_basis=7)),
            np.log,
        )
        np.testing.assert_allclose(data_basis, data_grid, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
