import skfda
from skfda.representation.basis import Monomial, Tensor, VectorValued
import unittest
import numpy as np


def ndm(*args):
    return [x[(None,) * i + (slice(None),) + (None,) * (len(args) - i - 1)]
            for i, x in enumerate(args)]


class InnerProductTest(unittest.TestCase):

    def test_several_variables(self):

        def f(x, y, z):
            return x * y * z

        t = np.linspace(0, 1, 100)

        x2, y2, z2 = ndm(t, 2 * t, 3 * t)

        data_matrix = f(x2, y2, z2)

        sample_points = [t, 2 * t, 3 * t]

        fd = skfda.FDataGrid(
            data_matrix[np.newaxis, ...], sample_points=sample_points)

        basis = Tensor([Monomial(n_basis=5, domain_range=(0, 1)),
                        Monomial(n_basis=5, domain_range=(0, 2)),
                        Monomial(n_basis=5, domain_range=(0, 3))])

        fd_basis = fd.to_basis(basis)

        res = 8

        np.testing.assert_allclose(
            skfda.misc.inner_product(fd, fd), res, rtol=1e-5)
        np.testing.assert_allclose(
            skfda.misc.inner_product(fd_basis, fd_basis), res, rtol=1e-5)

    def test_vector_valued(self):

        def f(x):
            return x**2

        def g(y):
            return 3 * y

        t = np.linspace(0, 1, 100)

        data_matrix = np.array([np.array([f(t), g(t)]).T])

        sample_points = [t]

        fd = skfda.FDataGrid(
            data_matrix, sample_points=sample_points)

        basis = VectorValued([Monomial(n_basis=5),
                              Monomial(n_basis=5)])

        fd_basis = fd.to_basis(basis)

        res = 1 / 5 + 3

        np.testing.assert_allclose(
            skfda.misc.inner_product(fd, fd), res, rtol=1e-5)
        np.testing.assert_allclose(
            skfda.misc.inner_product(fd_basis, fd_basis), res, rtol=1e-5)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
