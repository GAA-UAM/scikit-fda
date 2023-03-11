"""Tests of the LinearDifferentialOperator."""

import unittest
from typing import Callable, Sequence, Union

import numpy as np

from skfda.misc.operators import LinearDifferentialOperator
from skfda.representation.basis import ConstantBasis, FDataBasis, MonomialBasis

WeightCallable = Callable[[np.ndarray], np.ndarray]


class TestLinearDifferentialOperator(unittest.TestCase):
    """Tests of the linear differential operator."""

    def _assert_equal_weights(
        self,
        weights: Sequence[Union[float, WeightCallable]],
        weights2: Sequence[Union[float, WeightCallable]],
        msg: str,
    ) -> None:
        self.assertEqual(len(weights), len(weights2), msg)

        for w, w2 in zip(weights, weights2):

            eq = getattr(w, "equals", None)

            if eq is None:
                self.assertEqual(w, w2, msg)
            else:
                self.assertTrue(eq(w2), msg)

    def test_init_default(self) -> None:
        """Tests default initialization (do not penalize)."""
        lfd = LinearDifferentialOperator()
        weights = [0]

        self._assert_equal_weights(
            lfd.weights,
            weights,
            "Wrong list of weight functions of the linear operator",
        )

    def test_init_integer(self) -> None:
        """Tests initializations which only specify the order."""
        # Checks for a zero order Lfd object
        lfd_0 = LinearDifferentialOperator(order=0)
        weights = [1]

        self._assert_equal_weights(
            lfd_0.weights,
            weights,
            "Wrong list of weight functions of the linear operator",
        )

        # Checks for a non zero order Lfd object
        lfd_3 = LinearDifferentialOperator(3)
        weights = [0, 0, 0, 1]

        self._assert_equal_weights(
            lfd_3.weights,
            weights,
            "Wrong list of weight functions of the linear operator",
        )

        # Negative order must fail
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(-1)

    def test_init_list_int(self) -> None:
        """Tests initializations with integer weights."""
        weights = [1, 3, 4, 5, 6, 7]

        lfd = LinearDifferentialOperator(weights=weights)

        self._assert_equal_weights(
            lfd.weights,
            weights,
            "Wrong list of weight functions of the linear operator",
        )

    def test_init_list_fdatabasis(self) -> None:
        """Test initialization with functional weights."""
        n_basis = 4
        n_weights = 6

        monomial = MonomialBasis(domain_range=(0, 1), n_basis=n_basis)

        weights = np.arange(n_basis * n_weights).reshape((n_weights, n_basis))

        fd = FDataBasis(monomial, weights)

        fdlist = [FDataBasis(monomial, w) for w in weights]
        lfd = LinearDifferentialOperator(weights=fdlist)

        self._assert_equal_weights(
            lfd.weights,
            list(fd),
            "Wrong list of weight functions of the linear operator",
        )

        # Check failure if intervals do not match
        constant = ConstantBasis(domain_range=(0, 2))
        fdlist.append(FDataBasis(constant, 1))
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(weights=fdlist)

    def test_init_wrong_params(self) -> None:
        """Check invalid parameters."""
        # Check specifying both arguments fail
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(1, weights=[1, 1])

        # Check invalid domain range
        monomial = MonomialBasis(domain_range=(0, 1), n_basis=3)
        fdlist = [FDataBasis(monomial, [1, 2, 3])]

        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(
                weights=fdlist,
                domain_range=(0, 2),
            )


if __name__ == '__main__':
    unittest.main()
