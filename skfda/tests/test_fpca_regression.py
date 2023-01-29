"""Tests for FPCA."""
from __future__ import annotations

import unittest

import numpy as np

import skfda
from skfda.datasets import fetch_tecator
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import FPCARegression
from skfda.representation.basis import BSplineBasis


class FPCARegressionTestCase(unittest.TestCase):
    """Tests for principal component analysis."""

    def test_fpca_reg_against_fda_usc(self) -> None:
        """Check that the results obtained are similar to those of fda.usc.

        Results obtained from fda.usc with the following R code
            library("fda.usc")
            data(tecator)
            # Fit the regression model with the first 129 observations
            x=tecator$absorp.fdata[1:129,]
            y=tecator$y$Fat[1:129]
            res2=fregre.pc(x,y,l=1:10)

            # Predict the response for the remaining observations
            n = length(tecator$y$Fat)
            xnew=tecator$absorp.fdata[130:n,]
            result = predict(res2, xnew)
            names(result) = NULL

            # Output the predicted values
            paste(
                round(result,8),
                collapse = ", "
            )
        """
        X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
        X_train = X.iloc[:129, 0].values
        y_train = y["fat"][:129].values
        X_test = X.iloc[129:, 0].values

        fpca_regression = FPCARegression(n_components=10)

        fpca_regression.fit(X_train, y_train)
        r_predictions = np.array([  # noqa: WPS317
            24.64807256, 3.46507101, 42.8075275, 4.47183513, 5.15394474,
            8.70136459, 7.18140042, 12.26860992, 11.89761155, 11.43831823,
            21.70303956, 25.75704241, 30.7086502, 48.10692261, 10.6258519,
            4.07226986, 5.3564939, 7.34333975, 6.02196557, 7.17052614,
            8.96339574, 7.48137304, 8.80638685, 9.91950449, 9.61385587,
            10.2969699, 9.72478128, 10.02837667, 10.72087154, 15.50611227,
            13.75408906, 15.45072713, 15.77815535, 16.4328961, 21.82526526,
            25.98049031, 26.75486697, 25.45727954, 31.41793557, 28.88122152,
            29.93498123, 44.2845357, 54.8070113, 43.58988546, 18.35395886,
            6.77862943, 4.07226986, 9.00970654, 9.02405569, 8.25587234,
            12.1732419, 14.28128606, 16.82891438, 24.1552859, 29.6879001,
            34.1931269, 54.85800891, 11.0114267, 8.5458698, 7.80994054,
            6.58579322, 7.24964726, 8.85435095, 7.97581994, 7.85785756,
            7.83365218, 8.4137662, 9.33079262, 9.78579099, 10.19765504,
            11.24655925, 12.18472769, 13.15521698, 15.01021835,
            11.43831823, 16.05351555, 17.47985504, 19.86368787,
            23.8272112, 26.88801046, 26.79250389, 28.7660723,
            31.95784747, 34.81938152, 42.52653123, 54.85800891,
        ])

        predictions = fpca_regression.predict(X_test)

        np.testing.assert_allclose(r_predictions, predictions, rtol=5e-3)

    def test_fpca_reg_against_fda_usc_reg(self) -> None:
        """Check that the results obtained are similar to those of fda.usc.

        Results obtained from fda.usc with the following R code
            library("fda.usc")
            data(tecator)
            # Fit the regression model with the first 129 observations
            x=tecator$absorp.fdata[1:129,]
            y=tecator$y$Fat[1:129]
            res2=fregre.pc(x,y,l=1:10, lambda = 1, P=c(0,0,1))

            # Predict the response for the remaining observations
            n = length(tecator$y$Fat)
            xnew=tecator$absorp.fdata[130:n,]
            result = predict(res2, xnew)
            names(result) = NULL

            # Output the predicted values
            paste(
                round(result,8),
                collapse = ", "
            )
        """
        X, y = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=True)
        X_train = X.iloc[:129, 0].values
        y_train = y["fat"][:129].values
        X_test = X.iloc[129:, 0].values
        pen_order = 2

        # Two correction factors are needed to obtain the same results as
        # fda.usc

        # The first factor compensates for the fact that the difference
        # matrices in fda.usc are scaled by the mean of the deltas
        # between grid points. This diference is introduced in
        # the function D.penalty (fdata2pc.R:479) in fda.usc.
        grid_points = X_test[0].grid_points[0]
        grid_step = np.mean(np.diff(grid_points))
        factor1 = grid_step**(2 * pen_order - 1)

        # The second factor compensates a scaling done to the final
        # regularization matrix in fda.usc. This scaling is introduced
        # in the function fregre.pc (fregre.pc.R:165) in fda.usc.
        factor2 = np.diff(X_train.domain_range[0]) / len(grid_points)
        factor2 **= (2 * pen_order + 1)

        # The total factor that we need to apply to the regularization
        # parameter of the regression is the product of the two factors
        total_factor = factor1 * factor2

        fpca_regression = FPCARegression(
            n_components=10,
            pca_regularization=L2Regularization(
                LinearDifferentialOperator(pen_order),
                regularization_parameter=1.0,
            ),
            regression_regularization=L2Regularization(
                LinearDifferentialOperator(pen_order),
                regularization_parameter=1.0 * total_factor,  # noqa: WPS345
            ),
        )

        fpca_regression.fit(X_train, y_train)
        r_predictions = np.array([  # noqa: WPS317
            23.48391439, 8.45397463, 43.44474891, 6.0981306, 6.56256704,
            8.7670954, 6.22132003, 12.8571075, 14.11658575, 13.57439588,
            23.96133718, 26.81618552, 30.31754827, 47.34895476, 11.53618091,
            5.49448547, 5.60875954, 8.57410351, 6.41573393, 7.22929637,
            7.53334465, 6.68773565, 8.6818259, 10.74793989, 8.6782855,
            11.89552059, 10.40896226, 10.93569607, 11.87369452, 14.57111824,
            14.5522881, 13.64065941, 16.03304023, 15.60254571, 21.11275437,
            25.03334373, 26.57624209, 25.44579176, 31.43561996, 28.03967631,
            29.77583234, 41.44325753, 54.14277276, 44.0569423, 22.90209546,
            8.85428988, 5.49448547, 9.36725951, 7.86510101, 6.22786929,
            13.71579956, 15.88516382, 18.05432943, 23.13267508, 28.48278103,
            36.59999903, 54.07821032, 12.55490093, 9.85357179, 8.79073997,
            6.95827529, 8.00027917, 9.19686289, 9.35734081, 8.60987473,
            7.81116661, 8.57285562, 7.73267045, 10.19060339, 9.96290068,
            12.57528552, 11.601402, 12.22342486, 14.19036667, 13.57439588,
            16.14576329, 15.87859389, 19.76979144, 23.62449942, 26.47268082,
            26.14126515, 27.52545896, 30.58823007, 33.62623406, 40.96307478,
            54.07821032,
        ])

        predictions = fpca_regression.predict(X_test)

        # The tolerance is not lower because the results are not
        # exactly the same as in fda.usc. The components calculated
        # by fda.usc are not exactly the same (they are orthogonal).
        # Additionaly, the penalization matrices are calculated using
        # a different method.
        np.testing.assert_allclose(r_predictions, predictions, rtol=1.5e-2)

    def test_fpca_reg_basis_vs_grid(self):
        """
        Compare results between grid and basis.

        Compare that the end result is the same when using the grid or
        basis representation of the data.
        """
        X, y = fetch_tecator(return_X_y=True)
        X = X.to_basis(BSplineBasis(n_basis=20))
        X_train = X[:129]
        X_test = X[129:]

        sampling_grid = np.linspace(
            X.domain_range[0][0], X.domain_range[0][1], 100,
        )

        X_grid = X.to_grid(grid_points=sampling_grid)
        X_grid_train = X_grid[:129]
        X_grid_test = X_grid[129:]

        fpca_regression = FPCARegression(
            n_components=10,
            pca_regularization=L2Regularization(
                LinearDifferentialOperator(2), regularization_parameter=10,
            ),
            regression_regularization=L2Regularization(
                LinearDifferentialOperator(2), regularization_parameter=10,
            ),
        )

        fpca_regression.fit(X_train, y[:129, 0])
        predictions_basis = fpca_regression.predict(X_test)

        fpca_regression.fit(X_grid_train, y[:129, 0])
        predictions_grid = fpca_regression.predict(X_grid_test)

        np.testing.assert_allclose(
            predictions_basis,
            predictions_grid,
            rtol=6e-3,
        )


if __name__ == "__main__":
    unittest.main()
