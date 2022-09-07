"""Tests for FPCA."""
from __future__ import annotations
import unittest

import numpy as np

import skfda
from skfda.ml.regression import FPCARegression


class FPCARegressionTestCase(unittest.TestCase):
    """Tests for principal component analysis."""

    def test_fpca_reg_against_fda_usc(self) -> None:
        """Check that the results obtained are similar to those of fda.usc.

        Results obtained from fda.usc with the following R code
            library("fda.usc")
            data(tecator)
            absorp=tecator$absorp.fdata
            ind=1:129
            x=absorp[ind,]
            y=tecator$y$Fat[ind]
            res2=fregre.pc(x,y,l=c(1,2,3))
            summary(res2)
        The variability explained by 
        """
        X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
        X = X.iloc[:129, 0].values
        y = y["fat"][:129].values

        fpca_regression = FPCARegression(n_components=3)
        fpca_regression.fit(X, y)

        results = fpca_regression.explained_variance_ratio_

        expected_results = np.array(
            [0.9821, 0.0096, 0.0028],
        )

        np.testing.assert_allclose(results, expected_results, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
