"""Tests for ANOVA."""
import unittest

import numpy as np

from skfda.datasets import fetch_gait
from skfda.inference.anova import (
    oneway_anova,
    v_asymptotic_stat,
    v_sample_stat,
)
from skfda.representation import FDataGrid
from skfda.representation.basis import FourierBasis


class OnewayAnovaTests(unittest.TestCase):
    """Tests for ANOVA."""

    def test_oneway_anova_args(self) -> None:
        """Check behavior of test with invalid args."""
        with self.assertRaises(ValueError):
            oneway_anova(FDataGrid([0]), n_reps=-2)

    def test_v_stats_args(self) -> None:
        """Check behavior of statistic with invalid args."""
        with self.assertRaises(ValueError):
            v_sample_stat(FDataGrid([0]), [0, 1])
        with self.assertRaises(ValueError):
            v_asymptotic_stat(FDataGrid([0]), weights=[0, 1])
        with self.assertRaises(ValueError):
            v_asymptotic_stat(
                FDataGrid([[1, 1, 1], [1, 1, 1]]),
                weights=[0, 0],
            )

    def test_v_stats(self) -> None:
        """Test statistic behaviour."""
        n_features = 50
        weights = [1, 2, 3]
        t = np.linspace(0, 1, n_features)
        m1 = [1 for _ in range(n_features)]
        m2 = [2 for _ in range(n_features)]
        m3 = [3 for _ in range(n_features)]
        fd = FDataGrid([m1, m2, m3], grid_points=t)
        self.assertEqual(v_sample_stat(fd, weights), 7.0)
        self.assertAlmostEqual(
            v_sample_stat(
                fd.to_basis(FourierBasis(n_basis=5)),
                weights,
            ),
            7.0,
        )
        res = (
            (1 - 2 * np.sqrt(1 / 2)) ** 2
            + (1 - 3 * np.sqrt(1 / 3)) ** 2
            + (2 - 3 * np.sqrt(2 / 3)) ** 2
        )
        self.assertAlmostEqual(v_asymptotic_stat(fd, weights=weights), res)
        self.assertAlmostEqual(
            v_asymptotic_stat(
                fd.to_basis(FourierBasis(n_basis=5)),
                weights=weights,
            ),
            res,
        )

    def test_asymptotic_behaviour(self) -> None:
        """Test asymptotic behaviour."""
        dataset = fetch_gait()
        fd = dataset['data'].coordinates[1]
        fd1 = fd[:5]
        fd2 = fd[5:10]
        fd3 = fd[10:15]

        n_little_sim = 10

        sims = np.array([
            oneway_anova(
                fd1,
                fd2,
                fd3,
                n_reps=500,
                random_state=i,
            )[1]
            for i in range(n_little_sim)
        ])
        little_sim = np.mean(sims)
        big_sim = oneway_anova(fd1, fd2, fd3, n_reps=2000, random_state=100)[1]
        self.assertAlmostEqual(little_sim, big_sim, delta=0.05)

        fd = fd.to_basis(FourierBasis(n_basis=5))
        fd1 = fd[:5]
        fd2 = fd[5:10]

        sims = np.array([
            oneway_anova(
                fd1,
                fd2,
                n_reps=500,
                random_state=i,
            )[1]
            for i in range(n_little_sim)
        ])
        little_sim = np.mean(sims)
        big_sim = oneway_anova(fd1, fd2, n_reps=2000, random_state=100)[1]
        self.assertAlmostEqual(little_sim, big_sim, delta=0.05)


if __name__ == '__main__':
    unittest.main()
