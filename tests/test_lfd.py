import unittest

from fda.basis import Basis, FDataBasis, Constant, Monomial, BSpline, Fourier

from fda.lfd import Lfd

import numpy as np


class TestBasis(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_init_integer(self):
        # Checks for a zero order Lfd object
        lfd_0 = Lfd(0)

        np.testing.assert_equal(lfd_0.nderiv, 0, "Wrong deriv order of the linear operator")
        np.testing.assert_equal(lfd_0.bwtlist, [], "Wrong list of weight functions of the linear operator")

        # Checks for a non zero order Lfd object
        lfd_3 = Lfd(3)
        consfd = FDataBasis(Constant((0, 1)), 0)
        bwtlist3 = [consfd for _ in range(3)]

        np.testing.assert_equal(lfd_3.nderiv, 3, "Wrong deriv order of the linear operator")
        np.testing.assert_equal(lfd_3.bwtlist, bwtlist3, "Wrong list of weight functions of the linear operator")

        np.testing.assert_raises(ValueError, Lfd, -1)

    def test_init_fdatabasis(self):
        monomial = Monomial((0, 1), nbasis=2)
        fd = FDataBasis(monomial, np.arange(6).reshape(3, 2))
        lfd = Lfd(fd)

        np.testing.assert_equal(lfd.nderiv, 3, "Wrong deriv order of the linear operator")
        np.testing.assert_equal(lfd.bwtlist, fd.to_list(), "Wrong list of weight functions of the linear operator")


    def test_init_list_int(self):
        weights = [1, 3, 4, 5, 6, 7]

        constant = Constant((0,1))
        fd = FDataBasis(constant, np.array(weights).reshape(-1, 1))
        lfd = Lfd(weights)

        np.testing.assert_equal(lfd.nderiv, 6, "Wrong deriv order of the linear operator")
        np.testing.assert_equal(lfd.bwtlist, fd.to_list(), "Wrong list of weight functions of the linear operator")


    def test_init_list_fdatabasis(self):
        weights = np.arange(4 * 5).reshape((5, 4))
        monomial = Monomial((0, 1), nbasis=4)
        fd = FDataBasis(monomial, weights)

        fdlist = [FDataBasis(monomial, weights[i]) for i in range(len(weights))]

        lfd = Lfd(fdlist)

        a = str(lfd)

        np.testing.assert_equal(lfd.nderiv, 5, "Wrong deriv order of the linear operator")
        np.testing.assert_equal(lfd.bwtlist, fd.to_list(), "Wrong list of weight functions of the linear operator")

        contant = Constant((0,2))
        fdlist.append(FDataBasis(contant, 1))
        np.testing.assert_raises(ValueError, Lfd, fdlist)

    def test_init_wrong_params(self):
        np.testing.assert_raises(ValueError, Lfd, ['a'])
        np.testing.assert_raises(ValueError, Lfd, 'a')

if __name__ == '__main__':
    print()
    unittest.main()
