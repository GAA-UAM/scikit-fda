import unittest

import numpy as np

from skfda.representation.basis import FDataBasis, Constant, Monomial
from skfda.misc import LinearDifferentialOperator


class TestBasis(unittest.TestCase):

    def test_init_integer(self):
        # Checks for a zero order Lfd object
        lfd_0 = LinearDifferentialOperator(order=0)
        weightfd = [FDataBasis(Constant((0, 1)), 1)]

        np.testing.assert_equal(lfd_0.order, 0,
                                "Wrong deriv order of the linear operator")
        np.testing.assert_equal(
            lfd_0.weights, weightfd,
            "Wrong list of weight functions of the linear operator")

        # Checks for a non zero order Lfd object
        lfd_3 = LinearDifferentialOperator(3)
        consfd = FDataBasis(Constant((0, 1)), np.identity(4)[3].reshape(-1, 1))
        bwtlist3 = consfd.to_list()

        np.testing.assert_equal(lfd_3.order, 3,
                                "Wrong deriv order of the linear operator")
        np.testing.assert_equal(
            lfd_3.weights, bwtlist3,
            "Wrong list of weight functions of the linear operator")

        np.testing.assert_raises(ValueError, LinearDifferentialOperator, -1)

    def test_init_list_int(self):
        coefficients = [1, 3, 4, 5, 6, 7]

        constant = Constant((0, 1))
        fd = FDataBasis(constant, np.array(coefficients).reshape(-1, 1))
        lfd = LinearDifferentialOperator(weights=coefficients)

        np.testing.assert_equal(lfd.order, 5,
                                "Wrong deriv order of the linear operator")
        np.testing.assert_equal(
            lfd.weights, fd.to_list(),
            "Wrong list of weight functions of the linear operator")

    def test_init_list_fdatabasis(self):
        weights = np.arange(4 * 5).reshape((5, 4))
        monomial = Monomial((0, 1), nbasis=4)
        fd = FDataBasis(monomial, weights)

        fdlist = [FDataBasis(monomial, weights[i])
                  for i in range(len(weights))]

        lfd = LinearDifferentialOperator(weights=fdlist)

        np.testing.assert_equal(lfd.order, 4,
                                "Wrong deriv order of the linear operator")
        np.testing.assert_equal(
            lfd.weights, fd.to_list(),
            "Wrong list of weight functions of the linear operator")

        contant = Constant((0, 2))
        fdlist.append(FDataBasis(contant, 1))
        np.testing.assert_raises(ValueError, LinearDifferentialOperator,
                                 None, fdlist)

    def test_init_wrong_params(self):
        np.testing.assert_raises(ValueError,
                                 LinearDifferentialOperator, 0, ['a'])
        np.testing.assert_raises(ValueError,
                                 LinearDifferentialOperator, 0, 'a')


if __name__ == '__main__':
    print()
    unittest.main()
