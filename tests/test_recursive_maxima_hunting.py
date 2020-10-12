import skfda
from skfda.datasets import make_gaussian_process
from skfda.preprocessing.dim_reduction import variable_selection as vs
import unittest

import numpy as np


class TestRMH(unittest.TestCase):

    def test_rmh(self):
        n_samples = 10000
        n_features = 100

        def mean_1(t):
            return (np.abs(t - 0.25)
                    - 2 * np.abs(t - 0.5)
                    + np.abs(t - 0.75))

        X_0 = make_gaussian_process(n_samples=n_samples // 2,
                                    n_features=n_features,
                                    random_state=0)
        X_1 = make_gaussian_process(n_samples=n_samples // 2,
                                    n_features=n_features,
                                    mean=mean_1,
                                    random_state=1)
        X = skfda.concatenate((X_0, X_1))

        y = np.zeros(n_samples)
        y[n_samples // 2:] = 1

        correction = vs.recursive_maxima_hunting.GaussianSampleCorrection()
        stopping_condition = vs.recursive_maxima_hunting.ScoreThresholdStop(
            threshold=0.05)

        rmh = vs.RecursiveMaximaHunting(
            correction=correction,
            stopping_condition=stopping_condition)
        _ = rmh.fit(X, y)
        point_mask = rmh.get_support()
        points = X.grid_points[0][point_mask]
        np.testing.assert_allclose(points, [0.25, 0.5, 0.75], rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
