import unittest

import numpy as np

import skfda.misc.covariances


class TestsSklearn(unittest.TestCase):

    def setUp(self) -> None:
        unittest.TestCase.setUp(self)

        self.x = np.linspace(-1, 1, 1000)[:, np.newaxis]

    def _test_compare_sklearn(
        self,
        cov: skfda.misc.covariances.Covariance,
    ) -> None:
        cov_sklearn = cov.to_sklearn()
        cov_matrix = cov(self.x, self.x)
        cov_sklearn_matrix = cov_sklearn(self.x)

        np.testing.assert_array_almost_equal(cov_matrix, cov_sklearn_matrix)

    def test_linear(self) -> None:

        for variance in (1, 2):
            for intercept in (0, 1, 2):
                with self.subTest(variance=variance, intercept=intercept):
                    cov = skfda.misc.covariances.Linear(
                        variance=variance, intercept=intercept)
                    self._test_compare_sklearn(cov)

    def test_polynomial(self) -> None:

        # Test a couple of non-default parameters only for speed
        for variance in (2,):
            for intercept in (0, 2):
                for slope in (1, 2):
                    for degree in (1, 2, 3):
                        with self.subTest(
                            variance=variance,
                            intercept=intercept,
                            slope=slope,
                            degree=degree,
                        ):
                            cov = skfda.misc.covariances.Polynomial(
                                variance=variance,
                                intercept=intercept,
                                slope=slope,
                                degree=degree,
                            )
                            self._test_compare_sklearn(cov)

    def test_gaussian(self) -> None:

        for variance in (1, 2):
            for length_scale in (0.5, 1, 2):
                with self.subTest(
                    variance=variance,
                    length_scale=length_scale,
                ):
                    cov = skfda.misc.covariances.Gaussian(
                        variance=variance,
                        length_scale=length_scale,
                    )
                    self._test_compare_sklearn(cov)

    def test_exponential(self) -> None:

        for variance in (1, 2):
            for length_scale in (0.5, 1, 2):
                with self.subTest(
                    variance=variance,
                    length_scale=length_scale,
                ):
                    cov = skfda.misc.covariances.Exponential(
                        variance=variance,
                        length_scale=length_scale,
                    )
                    self._test_compare_sklearn(cov)

    def test_matern(self) -> None:

        # Test a couple of non-default parameters only for speed
        for variance in (2,):
            for length_scale in (0.5,):
                for nu in (0.5, 1, 1.5, 2.5, 3.5, np.inf):
                    with self.subTest(
                        variance=variance,
                        length_scale=length_scale,
                        nu=nu,
                    ):
                        cov = skfda.misc.covariances.Matern(
                            variance=variance,
                            length_scale=length_scale,
                            nu=nu,
                        )
                        self._test_compare_sklearn(cov)

    def test_white_noise(self) -> None:

        for variance in (1, 2):
            with self.subTest(variance=variance):
                cov = skfda.misc.covariances.WhiteNoise(variance=variance)
                self._test_compare_sklearn(cov)
