import skfda
from skfda._utils import _pairwise_commutative
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

        grid_points = [t, 2 * t, 3 * t]

        fd = skfda.FDataGrid(
            data_matrix[np.newaxis, ...], grid_points=grid_points)

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

        grid_points = [t]

        fd = skfda.FDataGrid(
            data_matrix, grid_points=grid_points)

        basis = VectorValued([Monomial(n_basis=5),
                              Monomial(n_basis=5)])

        fd_basis = fd.to_basis(basis)

        res = 1 / 5 + 3

        np.testing.assert_allclose(
            skfda.misc.inner_product(fd, fd), res, rtol=1e-5)
        np.testing.assert_allclose(
            skfda.misc.inner_product(fd_basis, fd_basis), res, rtol=1e-5)

    def test_matrix(self):

        basis = skfda.representation.basis.BSpline(n_basis=12)

        X = skfda.datasets.make_gaussian_process(
            n_samples=10, n_features=20,
            cov=skfda.misc.covariances.Gaussian(),
            random_state=0)
        Y = skfda.datasets.make_gaussian_process(
            n_samples=10, n_features=20,
            cov=skfda.misc.covariances.Gaussian(),
            random_state=1)

        X_basis = X.to_basis(basis)
        Y_basis = Y.to_basis(basis)

        gram = skfda.misc.inner_product_matrix(X, Y)
        gram_basis = skfda.misc.inner_product_matrix(X_basis, Y_basis)

        np.testing.assert_allclose(gram, gram_basis, rtol=1e-2)

        gram_pairwise = _pairwise_commutative(
            skfda.misc.inner_product, X, Y)

        np.testing.assert_allclose(gram, gram_pairwise)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
