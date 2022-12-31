"""Tests for Hotelling functions."""
import unittest

from skfda.inference.hotelling import hotelling_t2, hotelling_test_ind
from skfda.representation import FDataGrid
from skfda.representation.basis import FourierBasis


class HotellingTests(unittest.TestCase):
    """Tests for Hotelling statistic and test."""

    def test_hotelling_test_ind_args(self) -> None:
        """Test that invalid arguments are rejected in test."""
        fd1 = FDataGrid([[1, 1, 1]])

        with self.assertRaises(TypeError):
            hotelling_test_ind(fd1.to_basis(FourierBasis(n_basis=3)), fd1)
        with self.assertRaises(TypeError):
            hotelling_test_ind(fd1, fd1.to_basis(FourierBasis(n_basis=3)))
        with self.assertRaises(ValueError):
            hotelling_test_ind(fd1, fd1, n_reps=0)

    def test_hotelling_t2_args(self) -> None:
        """Test that invalid arguments are rejected in statistic."""
        fd1 = FDataGrid([[1, 1, 1]])

        with self.assertRaises(TypeError):
            hotelling_t2(fd1.to_basis(FourierBasis(n_basis=3)), fd1)
        with self.assertRaises(TypeError):
            hotelling_t2(fd1, fd1.to_basis(FourierBasis(n_basis=3)))

    def test_hotelling_t2(self) -> None:
        """Trivial checks for the statistic."""
        fd1 = FDataGrid([[1, 1, 1], [1, 1, 1]])
        fd2 = FDataGrid([[1, 1, 1], [2, 2, 2]])
        self.assertAlmostEqual(hotelling_t2(fd1, fd1), 0)
        self.assertAlmostEqual(hotelling_t2(fd1, fd2), 1)

        fd1 = fd1.to_basis(FourierBasis(n_basis=3))
        fd2 = fd2.to_basis(FourierBasis(n_basis=3))
        self.assertAlmostEqual(hotelling_t2(fd1, fd1), 0)
        self.assertAlmostEqual(hotelling_t2(fd1, fd2), 1)

    def test_hotelling_test(self) -> None:
        """Trivial checks for the test."""
        fd1 = FDataGrid([[1, 1, 1], [1, 1, 1]])
        fd2 = FDataGrid([[3, 3, 3], [2, 2, 2]])
        t2, pval, dist = hotelling_test_ind(
            fd1,
            fd2,
            return_dist=True,
            random_state=0,
        )
        self.assertAlmostEqual(t2, 9)
        self.assertAlmostEqual(pval, 0)
        self.assertEqual(len(dist), 6)
        reps = 5
        t2, pval, dist = hotelling_test_ind(
            fd1,
            fd2,
            return_dist=True,
            n_reps=reps,
            random_state=1,
        )
        self.assertEqual(len(dist), reps)


if __name__ == '__main__':
    unittest.main()
