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

        np.testing.assert_allclose(predictions, r_predictions, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
