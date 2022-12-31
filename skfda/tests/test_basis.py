"""Tests of basis functions."""

import itertools
import unittest

import numpy as np

import skfda
from skfda import concatenate
from skfda.datasets import fetch_weather
from skfda.misc import inner_product_matrix
from skfda.representation.basis import (
    BSplineBasis,
    ConstantBasis,
    FDataBasis,
    FourierBasis,
    MonomialBasis,
)
from skfda.representation.grid import FDataGrid


class TestBasis(unittest.TestCase):
    """Tests of basis and FDataBasis."""

    # def setUp(self): could be defined for set up before any test

    def test_from_data_cholesky(self) -> None:
        """Test basis conversion using Cholesky method."""
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSplineBasis((0, 1), n_basis=5)
        np.testing.assert_array_almost_equal(
            FDataBasis.from_data(
                x,
                grid_points=t,
                basis=basis,
                method='cholesky',
            ).coefficients.round(2),
            np.array([[1.0, 2.78, -3.0, -0.78, 1.0]]),
        )

    def test_from_data_qr(self) -> None:
        """Test basis conversion using QR method."""
        t = np.linspace(0, 1, 5)
        x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
        basis = BSplineBasis((0, 1), n_basis=5)
        np.testing.assert_array_almost_equal(
            FDataBasis.from_data(
                x,
                grid_points=t,
                basis=basis,
                method='qr',
            ).coefficients.round(2),
            np.array([[1.0, 2.78, -3.0, -0.78, 1.0]]),
        )

    def test_basis_inner_matrix(self) -> None:
        """Test the inner product matrix of FDataBasis objects."""
        basis = MonomialBasis(n_basis=3)

        np.testing.assert_array_almost_equal(
            basis.inner_product_matrix(),
            [
                [1, 1 / 2, 1 / 3],  # noqa: WPS204
                [1 / 2, 1 / 3, 1 / 4],  # noqa: WPS204
                [1 / 3, 1 / 4, 1 / 5],
            ],
        )

        np.testing.assert_array_almost_equal(
            basis.inner_product_matrix(basis),
            [
                [1, 1 / 2, 1 / 3],
                [1 / 2, 1 / 3, 1 / 4],
                [1 / 3, 1 / 4, 1 / 5],
            ],
        )

        np.testing.assert_array_almost_equal(
            basis.inner_product_matrix(MonomialBasis(n_basis=4)),
            [
                [1, 1 / 2, 1 / 3, 1 / 4],
                [1 / 2, 1 / 3, 1 / 4, 1 / 5],
                [1 / 3, 1 / 4, 1 / 5, 1 / 6],
            ],
        )

        # TODO testing with other basis

    def test_basis_gram_matrix_monomial(self) -> None:
        """Test the Gram matrix with monomial basis."""
        basis = MonomialBasis(n_basis=3)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()  # noqa: WPS437
        gram_matrix_res = np.array([
            [1, 1 / 2, 1 / 3],
            [1 / 2, 1 / 3, 1 / 4],
            [1 / 3, 1 / 4, 1 / 5],
        ])

        np.testing.assert_allclose(
            gram_matrix,
            gram_matrix_res,
        )
        np.testing.assert_allclose(
            gram_matrix_numerical,
            gram_matrix_res,
        )

    def test_basis_gram_matrix_fourier(self) -> None:
        """Test the Gram matrix with fourier basis."""
        basis = FourierBasis(n_basis=3)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()  # noqa: WPS437
        gram_matrix_res = np.identity(3)

        np.testing.assert_allclose(
            gram_matrix,
            gram_matrix_res,
        )
        np.testing.assert_allclose(
            gram_matrix_numerical,
            gram_matrix_res,
            atol=1e-15,
            rtol=1e-15,
        )

    def test_basis_gram_matrix_bspline(self) -> None:
        """Test the Gram matrix with B-spline basis."""
        basis = BSplineBasis(n_basis=6)
        gram_matrix = basis.gram_matrix()
        gram_matrix_numerical = basis._gram_matrix_numerical()  # noqa: WPS437
        gram_matrix_res = np.array([
            [0.04761905, 0.02916667, 0.00615079, 0.00039683, 0, 0],
            [0.02916667, 0.07380952, 0.05208333, 0.01145833, 0.00014881, 0],
            [  # noqa: WPS317
                0.00615079, 0.05208333, 0.10892857,
                0.07098214, 0.01145833, 0.00039683,
            ],
            [  # noqa: WPS317
                0.00039683, 0.01145833, 0.07098214,
                0.10892857, 0.05208333, 0.00615079,
            ],
            [0, 0.00014881, 0.01145833, 0.05208333, 0.07380952, 0.02916667],
            [0, 0, 0.00039683, 0.00615079, 0.02916667, 0.04761905],
        ])

        np.testing.assert_allclose(
            gram_matrix,
            gram_matrix_res,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            gram_matrix_numerical,
            gram_matrix_res,
            rtol=1e-4,
        )

    def test_basis_basis_inprod(self) -> None:
        """Test inner product between different basis."""
        monomial = MonomialBasis(n_basis=4)
        bspline = BSplineBasis(n_basis=5, order=4)
        np.testing.assert_allclose(
            monomial.inner_product_matrix(bspline),
            np.array([
                [0.12499983, 0.25000035, 0.24999965, 0.25000035, 0.12499983],
                [0.01249991, 0.07500017, 0.12499983, 0.17500017, 0.11249991],
                [0.00208338, 0.02916658, 0.07083342, 0.12916658, 0.10208338],
                [0.00044654, 0.01339264, 0.04375022, 0.09910693, 0.09330368],
            ]),
            rtol=1e-3,
        )
        np.testing.assert_array_almost_equal(
            monomial.inner_product_matrix(bspline),
            bspline.inner_product_matrix(monomial).T,
        )

    def test_basis_fdatabasis_inprod(self) -> None:
        """Test inner product between different basis expansions."""
        monomial = MonomialBasis(n_basis=4)
        bspline = BSplineBasis(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(monomial.to_basis(), bsplinefd),
            np.array([
                [2.0, 7.0, 12.0],
                [1.29626206, 3.79626206, 6.29626206],
                [0.96292873, 2.62959539, 4.29626206],
                [0.7682873, 2.0182873, 3.2682873],
            ]),
            rtol=1e-4,
        )

    def test_fdatabasis_fdatabasis_inprod(self) -> None:
        """Test inner product between FDataBasis objects."""
        monomial = MonomialBasis(n_basis=4)
        monomialfd = FDataBasis(
            monomial,
            [
                [5, 4, 1, 0],
                [4, 2, 1, 0],
                [4, 1, 6, 4],
                [4, 5, 0, 1],
                [5, 6, 2, 0],
            ],
        )
        bspline = BSplineBasis(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(monomialfd, bsplinefd),
            np.array([
                [16.14797697, 52.81464364, 89.4813103],
                [11.55565285, 38.22211951, 64.88878618],
                [18.14698361, 55.64698361, 93.14698361],
                [15.2495976, 48.9995976, 82.7495976],
                [19.70392982, 63.03676315, 106.37009648],
            ]),
            rtol=1e-4,
        )

    def test_comutativity_inprod(self) -> None:
        """Test commutativity of the inner product."""
        monomial = MonomialBasis(n_basis=4)
        bspline = BSplineBasis(n_basis=5, order=3)
        bsplinefd = FDataBasis(bspline, np.arange(0, 15).reshape(3, 5))

        np.testing.assert_allclose(
            inner_product_matrix(bsplinefd, monomial.to_basis()),
            np.transpose(inner_product_matrix(monomial.to_basis(), bsplinefd)),
        )

    def test_concatenate(self) -> None:
        """Test concatenation of two FDataBasis."""
        sample1 = np.arange(0, 10)
        sample2 = np.arange(10, 20)
        fd1 = FDataGrid([sample1]).to_basis(FourierBasis(n_basis=5))
        fd2 = FDataGrid([sample2]).to_basis(FourierBasis(n_basis=5))

        fd = concatenate([fd1, fd2])

        np.testing.assert_equal(fd.n_samples, 2)
        np.testing.assert_equal(fd.dim_codomain, 1)
        np.testing.assert_equal(fd.dim_domain, 1)
        np.testing.assert_array_equal(
            fd.coefficients,
            np.concatenate([fd1.coefficients, fd2.coefficients]),
        )


class TestFDataBasisOperations(unittest.TestCase):
    """Test FDataBasis operations."""

    def test_fdatabasis_add(self) -> None:
        """Test addition of FDataBasis."""
        monomial1 = FDataBasis(MonomialBasis(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(
            MonomialBasis(n_basis=3),
            [[1, 2, 3], [3, 4, 5]],
        )

        self.assertTrue(
            (monomial1 + monomial2).equals(
                FDataBasis(
                    MonomialBasis(n_basis=3),
                    [[2, 4, 6], [4, 6, 8]],
                ),
            ),
        )

        with np.testing.assert_raises(TypeError):
            monomial2 + FDataBasis(  # noqa: WPS428
                FourierBasis(n_basis=3),
                [[2, 2, 3], [5, 4, 5]],
            )

    def test_fdatabasis_sub(self) -> None:
        """Test subtraction of FDataBasis."""
        monomial1 = FDataBasis(MonomialBasis(n_basis=3), [1, 2, 3])
        monomial2 = FDataBasis(
            MonomialBasis(n_basis=3),
            [[1, 2, 3], [3, 4, 5]],
        )

        self.assertTrue(
            (monomial1 - monomial2).equals(
                FDataBasis(
                    MonomialBasis(n_basis=3),
                    [[0, 0, 0], [-2, -2, -2]],
                ),
            ),
        )

        with np.testing.assert_raises(TypeError):
            monomial2 - FDataBasis(  # noqa: WPS428
                FourierBasis(n_basis=3),
                [[2, 2, 3], [5, 4, 5]],
            )

    def test_fdatabasis_mul(self) -> None:
        """Test multiplication of FDataBasis."""
        basis = MonomialBasis(n_basis=3)

        monomial1 = FDataBasis(basis, [1, 2, 3])
        monomial2 = FDataBasis(basis, [[1, 2, 3], [3, 4, 5]])

        self.assertTrue(
            (monomial1 * 2).equals(
                FDataBasis(
                    basis,
                    [[2, 4, 6]],
                ),
            ),
        )

        self.assertTrue(
            (3 * monomial2).equals(
                FDataBasis(
                    basis,
                    [[3, 6, 9], [9, 12, 15]],
                ),
            ),
        )

        self.assertTrue(
            (3 * monomial2).equals(
                monomial2 * 3,
            ),
        )

        self.assertTrue(
            (monomial2 * np.array([1, 2])).equals(
                FDataBasis(
                    basis,
                    [[1, 2, 3], [6, 8, 10]],
                ),
            ),
        )

        self.assertTrue(
            (np.array([1, 2]) * monomial2).equals(
                FDataBasis(
                    basis,
                    [[1, 2, 3], [6, 8, 10]],
                ),
            ),
        )

        with np.testing.assert_raises(TypeError):
            monomial2 * FDataBasis(  # noqa: WPS428
                FourierBasis(n_basis=3),
                [[2, 2, 3], [5, 4, 5]],
            )

        with np.testing.assert_raises(TypeError):
            monomial2 * monomial2  # noqa: WPS428

    def test_fdatabasis_div(self) -> None:
        """Test division of FDataBasis."""
        basis = MonomialBasis(n_basis=3)

        monomial1 = FDataBasis(basis, [1, 2, 3])
        monomial2 = FDataBasis(basis, [[1, 2, 3], [3, 4, 5]])

        self.assertTrue((monomial1 / 2).equals(
            FDataBasis(
                basis,
                [[1 / 2, 1, 3 / 2]],
            ),
        ))

        self.assertTrue(
            (monomial2 / 2).equals(
                FDataBasis(
                    basis,
                    [[1 / 2, 1, 3 / 2], [3 / 2, 2, 5 / 2]],
                ),
            ),
        )

        self.assertTrue(
            (monomial2 / [1, 2]).equals(
                FDataBasis(
                    basis,
                    [[1.0, 2.0, 3.0], [3 / 2, 2, 5 / 2]],
                ),
            ),
        )


class TestFDataBasisDerivatives(unittest.TestCase):
    """Test FDataBasis derivatives."""

    def test_fdatabasis_derivative_constant(self) -> None:
        """Test derivatives with a constant basis."""
        constant = FDataBasis(
            ConstantBasis(),
            [[1], [2], [3], [4]],
        )

        self.assertTrue(
            constant.derivative().equals(
                FDataBasis(
                    ConstantBasis(),
                    [[0], [0], [0], [0]],
                ),
            ),
        )

        self.assertTrue(
            constant.derivative(order=0).equals(
                FDataBasis(
                    ConstantBasis(),
                    [[1], [2], [3], [4]],
                ),
            ),
        )

    def test_fdatabasis_derivative_monomial(self) -> None:
        """Test derivatives with a monomial basis."""
        monomial = FDataBasis(
            MonomialBasis(n_basis=8),
            [1, 5, 8, 9, 7, 8, 4, 5],
        )

        monomial2 = FDataBasis(
            MonomialBasis(n_basis=5),
            [
                [4, 9, 7, 4, 3],
                [1, 7, 9, 8, 5],
                [4, 6, 6, 6, 8],
            ],
        )

        self.assertTrue(
            monomial.derivative().equals(
                FDataBasis(
                    MonomialBasis(n_basis=7),
                    [5, 16, 27, 28, 40, 24, 35],
                ),
            ),
        )

        self.assertTrue(
            monomial.derivative(order=0).equals(monomial),
        )

        self.assertTrue(
            monomial.derivative(order=6).equals(
                FDataBasis(
                    MonomialBasis(n_basis=2),
                    [2880, 25200],
                ),
            ),
        )

        self.assertTrue(
            monomial2.derivative().equals(
                FDataBasis(
                    MonomialBasis(n_basis=4),
                    [
                        [9, 14, 12, 12],
                        [7, 18, 24, 20],
                        [6, 12, 18, 32],
                    ],
                ),
            ),
        )

        self.assertTrue(
            monomial2.derivative(order=0).equals(monomial2),
        )

        self.assertTrue(
            monomial2.derivative(order=3).equals(
                FDataBasis(
                    MonomialBasis(n_basis=2),
                    [
                        [24, 72],
                        [48, 120],
                        [36, 192],
                    ],
                ),
            ),
        )

    def test_fdatabasis_derivative_fourier(self) -> None:
        """Test derivatives with a fourier basis."""
        fourier = FDataBasis(
            FourierBasis(n_basis=7),
            [1, 5, 8, 9, 8, 4, 5],
        )

        fourier2 = FDataBasis(
            FourierBasis(n_basis=5),
            [
                [4, 9, 7, 4, 3],
                [1, 7, 9, 8, 5],
                [4, 6, 6, 6, 8],
            ],
        )

        fou0 = fourier.derivative(order=0)
        fou1 = fourier.derivative()
        fou2 = fourier.derivative(order=2)

        np.testing.assert_equal(fou1.basis, fourier.basis)
        np.testing.assert_almost_equal(
            fou1.coefficients.round(5),
            np.atleast_2d(
                [  # noqa: WPS317
                    0, -50.26548, 31.41593, -100.53096,
                    113.09734, -94.24778, 75.39822,
                ],
            ),
        )

        self.assertTrue(fou0.equals(fourier))
        np.testing.assert_equal(fou2.basis, fourier.basis)
        np.testing.assert_almost_equal(
            fou2.coefficients.round(5),
            np.atleast_2d(
                [  # noqa: WPS317
                    0, -197.39209, -315.82734, -1421.22303,
                    -1263.30936, -1421.22303, -1776.52879,
                ],
            ),
        )

        fou0 = fourier2.derivative(order=0)
        fou1 = fourier2.derivative()
        fou2 = fourier2.derivative(order=2)

        np.testing.assert_equal(fou1.basis, fourier2.basis)
        np.testing.assert_almost_equal(
            fou1.coefficients.round(5),
            [
                [0, -43.9823, 56.54867, -37.69911, 50.26548],
                [0, -56.54867, 43.9823, -62.83185, 100.53096],
                [0, -37.69911, 37.69911, -100.53096, 75.39822],
            ],
        )

        self.assertTrue(fou0.equals(fourier2))
        np.testing.assert_equal(fou2.basis, fourier2.basis)
        np.testing.assert_almost_equal(
            fou2.coefficients.round(5),
            [
                [0, -355.30576, -276.34892, -631.65468, -473.74101],
                [0, -276.34892, -355.30576, -1263.30936, -789.56835],
                [0, -236.87051, -236.87051, -947.48202, -1263.30936],
            ],
        )

    def test_fdatabasis_derivative_bspline(self) -> None:
        """Test derivatives with a B-spline basis."""
        bspline = FDataBasis(
            BSplineBasis(n_basis=8),
            [1, 5, 8, 9, 7, 8, 4, 5],
        )
        bspline2 = FDataBasis(
            BSplineBasis(n_basis=5),
            [
                [4, 9, 7, 4, 3],
                [1, 7, 9, 8, 5],
                [4, 6, 6, 6, 8],
            ],
        )

        bs0 = bspline.derivative(order=0)
        bs1 = bspline.derivative()
        bs2 = bspline.derivative(order=2)
        np.testing.assert_equal(bs1.basis, BSplineBasis(n_basis=7, order=3))

        np.testing.assert_almost_equal(
            bs1.coefficients,
            np.atleast_2d([60, 22.5, 5, -10, 5, -30, 15]),
        )

        self.assertTrue(bs0.equals(bspline))

        np.testing.assert_equal(
            bs2.basis,
            BSplineBasis(n_basis=6, order=2),
        )

        np.testing.assert_almost_equal(
            bs2.coefficients,
            np.atleast_2d([-375, -87.5, -75, 75, -175, 450]),
        )

        bs0 = bspline2.derivative(order=0)
        bs1 = bspline2.derivative()
        bs2 = bspline2.derivative(order=2)

        np.testing.assert_equal(bs1.basis, BSplineBasis(n_basis=4, order=3))

        np.testing.assert_almost_equal(
            bs1.coefficients,
            [
                [30, -6, -9, -6],
                [36, 6, -3, -18],
                [12, 0, 0, 12],
            ],
        )

        self.assertTrue(bs0.equals(bspline2))

        np.testing.assert_equal(
            bs2.basis,
            BSplineBasis(n_basis=3, order=2),
        )

        np.testing.assert_almost_equal(
            bs2.coefficients,
            [
                [-144, -6, 12],
                [-120, -18, -60],
                [-48, 0, 48],
            ],
        )


class TestVectorValuedBasis(unittest.TestCase):
    """Tests for the vector valued basis."""

    def test_vector_valued(self) -> None:
        """Test vector valued basis."""
        X, _ = fetch_weather(return_X_y=True)

        basis_dim = skfda.representation.basis.FourierBasis(
            n_basis=7,
            domain_range=X.domain_range,
        )
        basis = skfda.representation.basis.VectorValuedBasis(
            [basis_dim] * 2,
        )

        X_basis = X.to_basis(basis)

        self.assertEqual(X_basis.dim_codomain, 2)

        self.assertEqual(X_basis.coordinates[0].basis, basis_dim)
        np.testing.assert_allclose(
            X_basis.coordinates[0].coefficients,
            X.coordinates[0].to_basis(basis_dim).coefficients,
        )

        self.assertEqual(X_basis.coordinates[1].basis, basis_dim)
        np.testing.assert_allclose(
            X_basis.coordinates[1].coefficients,
            X.coordinates[1].to_basis(basis_dim).coefficients,
        )


class TestTensorBasis(unittest.TestCase):
    """Tests for the Tensor basis."""

    def setUp(self) -> None:
        """Create original and tensor bases."""
        self.n_x = 4
        self.n_y = 3
        self.n_z = 5

        self.n = self.n_x * self.n_y * self.n_z

        self.dims = (self.n_x, self.n_y, self.n_z)

        self.basis_x = skfda.representation.basis.MonomialBasis(
            n_basis=self.n_x,
        )
        self.basis_y = skfda.representation.basis.FourierBasis(
            n_basis=self.n_y,
        )
        self.basis_z = skfda.representation.basis.BSplineBasis(
            n_basis=self.n_z,
        )

        self.basis = skfda.representation.basis.TensorBasis([
            self.basis_x,
            self.basis_y,
            self.basis_z,
        ])

    def test_tensor_order(self) -> None:
        """
        Check the order of the elements in the tensor basis.

        The order should be:

        a_1 b_1 c_1, a_1 b_1 c_2, ..., a_1 b_1 c_n,
        a_1 b_2 c_1, a_1 b_1 c_2, ..., a_1 b_2 c_n,
        .
        .
        .
        a_1 b_m c_1, a_1 b_1 c_2, ..., a_1 b_m c_n,
        a_2 b_1 c_1, a_2 b_1 c_2, ..., a_2 b_1 c_n,
        .
        .
        .

        where the bases of the original spaces are A, B and C.

        """
        x_vals = [0, 0.3, 0.7]
        y_vals = [0.2, 0.5, 0.9]
        z_vals = [0.1, 0.4, 0.8]

        for t in itertools.product(x_vals, y_vals, z_vals):

            val_x = self.basis_x(t[0])
            val_y = self.basis_y(t[1])
            val_z = self.basis_z(t[2])
            val = self.basis([t])

            for x, y, z in itertools.product(
                range(self.n_x),
                range(self.n_y),
                range(self.n_z),
            ):

                index = (
                    x * self.n_y * self.n_z
                    + y * self.n_z
                    + z
                )

                index2 = np.ravel_multi_index(
                    [x, y, z],
                    dims=self.dims,
                )

                self.assertEqual(index, index2)

                self.assertAlmostEqual(
                    val[index],
                    val_x[x] * val_y[y] * val_z[z],
                )

    def test_tensor_gram_matrix(self) -> None:
        """Check that the Gram matrix is right."""
        gram_x = self.basis_x.gram_matrix()
        gram_y = self.basis_y.gram_matrix()
        gram_z = self.basis_z.gram_matrix()

        gram = self.basis.gram_matrix()

        for i in range(self.n):
            for j in range(self.n):
                left = np.unravel_index(i, shape=self.dims)
                right = np.unravel_index(j, shape=self.dims)

                value_gram = gram[i, j]
                value_gram_x = gram_x[left[0], right[0]]
                value_gram_y = gram_y[left[1], right[1]]
                value_gram_z = gram_z[left[2], right[2]]

                self.assertAlmostEqual(
                    value_gram,
                    value_gram_x * value_gram_y * value_gram_z,
                )


if __name__ == '__main__':
    unittest.main()
