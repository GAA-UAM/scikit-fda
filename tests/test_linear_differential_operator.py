from skfda.misc.operators import LinearDifferentialOperator
from skfda.representation.basis import FDataBasis, Constant, Monomial
import unittest

import numpy as np


class TestLinearDifferentialOperator(unittest.TestCase):

    def test_init_default(self):
        """Tests default initialization (do not penalize)."""
        lfd = LinearDifferentialOperator()
        weightfd = [FDataBasis(Constant((0, 1)), 0)]

        np.testing.assert_equal(
            lfd.weights, weightfd,
            "Wrong list of weight functions of the linear operator")

    def test_init_integer(self):
        """Tests initializations which only specify the order."""

        # Checks for a zero order Lfd object
        lfd_0 = LinearDifferentialOperator(order=0)
        weightfd = [FDataBasis(Constant((0, 1)), 1)]

        np.testing.assert_equal(
            lfd_0.weights, weightfd,
            "Wrong list of weight functions of the linear operator")

        # Checks for a non zero order Lfd object
        lfd_3 = LinearDifferentialOperator(3)
        consfd = FDataBasis(Constant((0, 1)), [[0], [0], [0], [1]])
        bwtlist3 = list(consfd)

        np.testing.assert_equal(
            lfd_3.weights, bwtlist3,
            "Wrong list of weight functions of the linear operator")

        # Negative order must fail
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(-1)

    def test_init_list_int(self):
        """Tests initializations with integer weights."""

        coefficients = [1, 3, 4, 5, 6, 7]

        constant = Constant((0, 1))
        fd = FDataBasis(constant, np.array(coefficients).reshape(-1, 1))

        lfd = LinearDifferentialOperator(weights=coefficients)

        np.testing.assert_equal(
            lfd.weights, list(fd),
            "Wrong list of weight functions of the linear operator")

    def test_init_list_fdatabasis(self):
        """Test initialization with functional weights."""

        n_basis = 4
        n_weights = 6

        monomial = Monomial((0, 1), n_basis=n_basis)

        weights = np.arange(n_basis * n_weights).reshape((n_weights, n_basis))

        fd = FDataBasis(monomial, weights)

        fdlist = [FDataBasis(monomial, w) for w in weights]
        lfd = LinearDifferentialOperator(weights=fdlist)

        np.testing.assert_equal(
            lfd.weights, list(fd),
            "Wrong list of weight functions of the linear operator")

        # Check failure if intervals do not match
        constant = Constant((0, 2))
        fdlist.append(FDataBasis(constant, 1))
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(weights=fdlist)

    def test_init_wrong_params(self):

        # Check specifying both arguments fail
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(1, weights=[1, 1])

        # Check invalid domain range
        monomial = Monomial((0, 1), n_basis=3)
        fdlist = [FDataBasis(monomial, [1, 2, 3])]

        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(weights=fdlist,
                                       domain_range=(0, 2))

        # Check wrong types fail
        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(weights=['a'])

        with np.testing.assert_raises(ValueError):
            LinearDifferentialOperator(weights='a')


if __name__ == '__main__':
    print()
    unittest.main()
