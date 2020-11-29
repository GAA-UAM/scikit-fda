import skfda
from skfda.exploratory.stats import geometric_median
import unittest
import numpy as np


class TestGeometricMedian(unittest.TestCase):

    def test_R_comparison(self):
        """
        Compare the results obtained using a real-world dataset with those in
        R (Gmedian package).

        """

        X, _ = skfda.datasets.fetch_tecator(return_X_y=True)

        r_res = [2.74083,  2.742715, 2.744627, 2.74659,  2.748656,
                 2.750879, 2.753307, 2.755984, 2.758927, 2.762182,
                 2.765724, 2.76957,  2.773756, 2.778333, 2.783346,
                 2.788818, 2.794758, 2.801225, 2.808233, 2.815714,
                 2.82351,  2.831355, 2.838997, 2.846298, 2.853295,
                 2.860186, 2.867332, 2.875107, 2.883778, 2.893419,
                 2.903851, 2.914717, 2.925698, 2.936765, 2.948293,
                 2.960908, 2.97526,  2.991206, 3.008222, 3.02552,
                 3.042172, 3.057356, 3.070666, 3.082351, 3.093396,
                 3.105338, 3.119946, 3.139307, 3.164418, 3.196014,
                 3.234248, 3.278306, 3.326051, 3.374015, 3.418148,
                 3.455051, 3.483095, 3.502789, 3.515961, 3.524557,
                 3.530135, 3.53364,  3.535369, 3.535305, 3.533326,
                 3.529343, 3.523357, 3.51548,  3.5059,   3.494807,
                 3.482358, 3.468695, 3.453939, 3.438202, 3.421574,
                 3.404169, 3.386148, 3.367751, 3.349166, 3.330441,
                 3.311532, 3.292318, 3.272683, 3.252482, 3.23157,
                 3.2099,   3.187632, 3.165129, 3.14282,  3.121008,
                 3.099793, 3.079092, 3.058772, 3.038755, 3.019038,
                 2.99963,  2.980476, 2.961467, 2.94252,  2.923682]

        median = geometric_median(X)
        median_multivariate = geometric_median(X.data_matrix[..., 0])

        np.testing.assert_allclose(
            median.data_matrix[0, :, 0], median_multivariate, rtol=1e-4)

        np.testing.assert_allclose(median_multivariate, r_res, rtol=1e-6)
