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

        r_res = [2.750514, 2.752771, 2.755024, 2.75733,  2.759735, 2.762285,
                 2.76502,  2.767978, 2.771194, 2.774686, 2.778441, 2.782469,
                 2.786811, 2.791514, 2.796613, 2.802125, 2.808058, 2.814479,
                 2.821395, 2.828766, 2.836444, 2.844187, 2.851768, 2.859055,
                 2.86607,  2.872991, 2.880089, 2.887727, 2.896155, 2.905454,
                 2.915467, 2.925852, 2.936333, 2.946924, 2.95798,  2.970123,
                 2.983961, 2.999372, 3.015869, 3.032706, 3.049022, 3.064058,
                 3.077409, 3.089294, 3.100633, 3.112871, 3.127676, 3.147024,
                 3.171922, 3.203067, 3.240606, 3.283713, 3.330258, 3.376808,
                 3.41942,  3.454856, 3.481628, 3.500368, 3.512892, 3.521134,
                 3.526557, 3.530016, 3.531786, 3.531848, 3.530082, 3.526385,
                 3.520757, 3.513308, 3.504218, 3.493666, 3.481803, 3.468755,
                 3.454654, 3.439589, 3.423664, 3.406963, 3.389647, 3.371963,
                 3.354073, 3.336043, 3.317809, 3.299259, 3.280295, 3.260775,
                 3.240553, 3.219589, 3.198045, 3.176265, 3.15465,  3.133493,
                 3.112882, 3.09274,  3.072943, 3.053437, 3.034223, 3.015319,
                 2.996664, 2.978161, 2.959728, 2.941405]

        #median = geometric_median(X)
        #median_multivariate = geometric_median(X.data_matrix[..., 0])

        np.testing.assert_allclose(
            median.data_matrix[0, :, 0], median_multivariate, rtol=1e-5)

        np.testing.assert_allclose(median_multivariate, r_res)
