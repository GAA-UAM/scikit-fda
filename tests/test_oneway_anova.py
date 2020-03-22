import unittest
import numpy as np

from skfda.representation import FDataGrid
from skfda.datasets import make_gaussian_process, fetch_gait
from skfda.inference.anova import  oneway_anova, v_asymptotic_stat, \
    v_sample_stat


class OnewayAnovaTests(unittest.TestCase):

    def test_oneway_anova_args(self):
        with self.assertRaises(ValueError):
            oneway_anova()
        with self.assertRaises(ValueError):
            oneway_anova(1, '2')
        with self.assertRaises(ValueError):
            oneway_anova(FDataGrid([0]), n_sim=-2)

    def test_v_stats_args(self):
        with self.assertRaises(ValueError):
            v_sample_stat(1, [1])
        with self.assertRaises(ValueError):
            v_sample_stat(FDataGrid([0]), [0, 1])
        with self.assertRaises(ValueError):
            v_asymptotic_stat(1, [1])
        with self.assertRaises(ValueError):
            v_asymptotic_stat(FDataGrid([0]), [0, 1])

    def test_v_stats(self):
        n_features = 50
        weights = [1, 2, 3]
        t = np.linspace(0, 1, n_features)
        m1 = [1 for _ in range(n_features)]
        m2 = [2 for _ in range(n_features)]
        m3 = [3 for _ in range(n_features)]
        fd = FDataGrid([m1, m2, m3], sample_points=t)
        self.assertEqual(v_sample_stat(fd, weights), 7.0)
        self.assertEqual(v_sample_stat(fd, weights, p=1), 5.0)
        res = (1 - 2 * np.sqrt(1 / 2)) ** 2 + (1 - 3 * np.sqrt(1 / 3)) ** 2 \
                                            + (2 - 3 * np.sqrt(2 / 3)) ** 2
        self.assertAlmostEqual(v_asymptotic_stat(fd, weights), res)
        res = abs(1 - 2 * np.sqrt(1 / 2)) + abs(1 - 3 * np.sqrt(1 / 3))\
                                          + abs(2 - 3 * np.sqrt(2 / 3))
        self.assertAlmostEqual(v_asymptotic_stat(fd, weights, p=1), res)

    def test_asymptotic_behaviour(self):
        dataset = fetch_gait()
        fd = dataset['data'].coordinates[1]
        fd1 = fd[0:13]
        fd2 = fd[13:26]
        fd3 = fd[26:39]

        n_little_sim = 50

        sims = np.array([oneway_anova(fd1, fd2, fd3, n_sim=2000)[1] for _ in
                         range(n_little_sim)])
        little_sim = np.mean(sims)
        big_sim = oneway_anova(fd1, fd2, fd3, n_sim=50000)[1]
        self.assertAlmostEqual(little_sim, big_sim, delta=0.01)


if __name__ == '__main__':
    print()
    unittest.main()
