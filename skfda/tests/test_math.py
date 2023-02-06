"""Test the math module."""
import unittest
from typing import Sequence

import numpy as np

import skfda
from skfda._utils import _pairwise_symmetric
from skfda.datasets import make_gaussian_process
from skfda.misc.covariances import Gaussian
from skfda.representation.basis import (
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)


def _ndm(
    *args: np.typing.NDArray[np.float_],
) -> Sequence[np.typing.NDArray[np.float_]]:
    return [
        x[(None,) * i + (slice(None),) + (None,) * (len(args) - i - 1)]
        for i, x in enumerate(args)
    ]


class InnerProductTest(unittest.TestCase):
    """Tests for the inner product."""

    def test_several_variables(self) -> None:
        """Test inner_product with functions of several variables."""
        def f(  # noqa: WPS430
            x: np.typing.NDArray[np.float_],
            y: np.typing.NDArray[np.float_],
            z: np.typing.NDArray[np.float_],
        ) -> np.typing.NDArray[np.float_]:
            return x * y * z

        t = np.linspace(0, 1, 30)

        x2, y2, z2 = _ndm(t, 2 * t, 3 * t)

        data_matrix = f(x2, y2, z2)

        grid_points = [t, 2 * t, 3 * t]

        fd = skfda.FDataGrid(
            data_matrix[np.newaxis, ...],
            grid_points=grid_points,
        )

        basis = TensorBasis([
            MonomialBasis(n_basis=5, domain_range=(0, 1)),
            MonomialBasis(n_basis=5, domain_range=(0, 2)),
            MonomialBasis(n_basis=5, domain_range=(0, 3)),
        ])

        fd_basis = fd.to_basis(basis)

        res = 8

        np.testing.assert_allclose(
            skfda.misc.inner_product(fd, fd),
            res,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            skfda.misc.inner_product(fd_basis, fd_basis),
            res,
            rtol=1e-4,
        )

    def test_vector_valued(self) -> None:
        """Test inner_product with vector valued functions."""
        def f(  # noqa: WPS430
            x: np.typing.NDArray[np.float_],
        ) -> np.typing.NDArray[np.float_]:
            return x**2

        def g(  # noqa: WPS430
            y: np.typing.NDArray[np.float_],
        ) -> np.typing.NDArray[np.float_]:
            return 3 * y

        t = np.linspace(0, 1, 100)

        data_matrix = np.array([np.array([f(t), g(t)]).T])

        grid_points = [t]

        fd = skfda.FDataGrid(
            data_matrix,
            grid_points=grid_points,
        )

        basis = VectorValuedBasis([
            MonomialBasis(n_basis=5),
            MonomialBasis(n_basis=5),
        ])

        fd_basis = fd.to_basis(basis)

        res = 1 / 5 + 3

        np.testing.assert_allclose(
            skfda.misc.inner_product(fd, fd),
            res,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            skfda.misc.inner_product(fd_basis, fd_basis),
            res,
            rtol=1e-5,
        )

    def test_matrix(self) -> None:
        """Test inner_product_matrix function."""
        basis = skfda.representation.basis.BSplineBasis(n_basis=12)

        X = make_gaussian_process(
            n_samples=10,
            n_features=20,
            cov=Gaussian(),
            random_state=0,
        )
        Y = make_gaussian_process(
            n_samples=10,
            n_features=20,
            cov=Gaussian(),
            random_state=1,
        )

        X_basis = X.to_basis(basis)
        Y_basis = Y.to_basis(basis)

        gram = skfda.misc.inner_product_matrix(X, Y)
        gram_basis = skfda.misc.inner_product_matrix(X_basis, Y_basis)

        np.testing.assert_allclose(gram, gram_basis, rtol=1e-2)

        gram_pairwise = _pairwise_symmetric(
            skfda.misc.inner_product,
            X,
            Y,
        )

        np.testing.assert_allclose(gram, gram_pairwise)


class CosineSimilarityVectorTest(unittest.TestCase):
    """Tests for cosine similarity for vectors."""

    def setUp(self) -> None:
        """Create examples."""
        self.arr = np.array([
            [0, 0, 1],
            [1, 1, 1],
            [1, 2, 3],
            [1, 0, 1],
        ])

        self.arr_samelen = np.array([
            [2, 4, 1],
            [7, 2, 1],
            [0, 1, 0],
            [3, 2, 0],
        ])

        self.arr_short = np.array([
            [2, 4, 6],
            [5, 1, 7],
        ])

    def test_cosine_similarity_elementwise(self) -> None:
        """Elementwise example for vectors."""
        np.testing.assert_allclose(
            skfda.misc.inner_product(self.arr, self.arr_samelen),
            [1, 10, 2, 3],
        )

        np.testing.assert_allclose(
            skfda.misc.cosine_similarity(self.arr, self.arr_samelen),
            [
                1 / np.sqrt(21),
                10 / np.sqrt(3) / np.sqrt(54),
                2 / np.sqrt(14),
                3 / np.sqrt(2) / np.sqrt(13),
            ],
        )

    def test_cosine_similarity_matrix_one(self) -> None:
        """Matrix example for vectors with one input."""
        for arr2 in (None, self.arr):

            np.testing.assert_allclose(
                skfda.misc.inner_product_matrix(self.arr, arr2),
                [
                    [1, 1, 3, 1],
                    [1, 3, 6, 2],
                    [3, 6, 14, 4],
                    [1, 2, 4, 2],
                ],
            )

            np.testing.assert_allclose(
                skfda.misc.cosine_similarity_matrix(self.arr, arr2),
                [
                    [
                        1,
                        1 / np.sqrt(3),
                        3 / np.sqrt(14),
                        1 / np.sqrt(2),
                    ],
                    [
                        1 / np.sqrt(3),
                        3 / np.sqrt(3 * 3),
                        6 / np.sqrt(3 * 14),
                        2 / np.sqrt(3 * 2),
                    ],
                    [
                        3 / np.sqrt(14),
                        6 / np.sqrt(14 * 3),
                        14 / np.sqrt(14 * 14),
                        4 / np.sqrt(14 * 2),
                    ],
                    [
                        1 / np.sqrt(2),
                        2 / np.sqrt(2 * 3),
                        4 / np.sqrt(2 * 14),
                        2 / np.sqrt(2 * 2),
                    ],
                ],
            )

    def test_cosine_similarity_matrix_two(self) -> None:
        """Matrix example for vectors with two inputs."""
        np.testing.assert_allclose(
            skfda.misc.inner_product_matrix(self.arr, self.arr_short),
            [
                [6, 7],
                [12, 13],
                [28, 28],
                [8, 12],
            ],
        )

        np.testing.assert_allclose(
            skfda.misc.cosine_similarity_matrix(self.arr, self.arr_short),
            [
                [
                    6 / np.sqrt(56),
                    7 / np.sqrt(75),
                ],
                [
                    12 / np.sqrt(3) / np.sqrt(56),
                    13 / np.sqrt(3) / np.sqrt(75),
                ],
                [
                    28 / np.sqrt(14) / np.sqrt(56),
                    28 / np.sqrt(14) / np.sqrt(75),
                ],
                [
                    8 / np.sqrt(2) / np.sqrt(56),
                    12 / np.sqrt(2) / np.sqrt(75),
                ],
            ],
        )


if __name__ == "__main__":
    unittest.main()
