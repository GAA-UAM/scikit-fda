"""Tests for FDataBasis evaluation."""
import unittest

import numpy as np

from skfda.representation.basis import (
    BSplineBasis,
    ConstantBasis,
    FDataBasis,
    FourierBasis,
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)


class TestFDataBasisEvaluation(unittest.TestCase):
    """
    Test FDataBasis evaluation.

    This class includes tests that don't depend on the particular based used.

    """

    def test_evaluation_single_point(self) -> None:
        """Test the evaluation of a single point FDataBasis."""
        fourier = FourierBasis(
            domain_range=(0, 1),
            n_basis=3,
        )

        coefficients = np.array([
            [0.00078238, 0.48857741, 0.63971985],
            [0.01778079, 0.73440271, 0.20148638],
        ])

        f = FDataBasis(fourier, coefficients)

        # Test different ways of call f with a point
        res = np.array(
            [-0.903918107989282, -0.267163981229459],
        ).reshape((2, 1, 1))

        np.testing.assert_allclose(f([0.5]), res)
        np.testing.assert_allclose(f((0.5,)), res)
        np.testing.assert_allclose(f(0.5), res)
        np.testing.assert_allclose(f(np.array([0.5])), res)
        np.testing.assert_allclose(f(np.array(0.5)), res)

    def test_evaluation_grid_1d(self) -> None:
        """
        Test the evaluation of FDataBasis with grid=True.

        Nothing should change as the domain dimension is 1.

        """
        fourier = FourierBasis(
            domain_range=(0, 1),
            n_basis=3,
        )

        coefficients = np.array([
            [0.00078238, 0.48857741, 0.63971985],
            [0.01778079, 0.73440271, 0.20148638],
        ])

        f = FDataBasis(fourier, coefficients)
        t = np.linspace(0, 1, 4)

        res_test = f(t)

        # Different ways to pass the axes
        np.testing.assert_allclose(
            f(t, grid=True),
            res_test,
        )
        np.testing.assert_allclose(
            f((t,), grid=True),
            res_test,
        )
        np.testing.assert_allclose(
            f([t], grid=True),
            res_test,
        )
        np.testing.assert_allclose(
            f(np.atleast_2d(t), grid=True),
            res_test,
        )

        # Number of axis different than the domain dimension (1)
        with np.testing.assert_raises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_unaligned(self) -> None:
        """Test the unaligned evaluation."""
        fourier = FourierBasis(
            domain_range=(0, 1),
            n_basis=3,
        )

        coefficients = np.array([
            [0.00078238, 0.48857741, 0.63971985],
            [0.01778079, 0.73440271, 0.20148638],
        ])

        f = FDataBasis(fourier, coefficients)
        t = np.linspace(0, 1, 4)

        # Test same result than normal evaluation
        np.testing.assert_allclose(
            f([1]),
            f([[1], [1]], aligned=False),
        )
        np.testing.assert_allclose(
            f(t),
            f(np.vstack((t, t)), aligned=False),
        )

        # Different evaluation times
        t_multiple = [[0, 0.5], [0.2, 0.7]]
        np.testing.assert_allclose(
            f(t_multiple[0])[0],
            f(t_multiple, aligned=False)[0],
        )

        np.testing.assert_allclose(
            f(t_multiple[1])[1],
            f(t_multiple, aligned=False)[1],
        )


class TestBasisEvaluationFourier(unittest.TestCase):
    """Test FDataBasis with Fourier basis."""

    def test_simple_evaluation(self) -> None:
        """Compare Fourier evaluation with R package fda."""
        fourier = FourierBasis(
            domain_range=(0, 2),
            n_basis=5,
        )

        coefficients = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array([
            [  # noqa: WPS317
                8.71, 9.66, 1.84, -4.71, -2.80, 2.71,
                2.45, -3.82, -6.66, -0.30, 8.71,
            ],
            [  # noqa: WPS317
                22.24, 26.48, 10.57, -4.95, -3.58, 6.24,
                5.31, -7.69, -13.32, 1.13, 22.24,
            ],
        ])[..., np.newaxis]

        np.testing.assert_allclose(f(t), res, atol=1e-2)

    def test_evaluation_derivative(self) -> None:
        """Test the evaluation of the derivative of Fourier."""
        fourier = FourierBasis(domain_range=(0, 1), n_basis=3)

        coefficients = np.array([
            [0.00078238, 0.48857741, 0.63971985],
            [0.01778079, 0.73440271, 0.20148638],
        ])

        f = FDataBasis(fourier, coefficients)

        t = np.linspace(0, 1, 4)

        res = np.array([  # noqa: WPS317
            4.34138447771721, -7.09352774867064, 2.75214327095343,
            4.34138447771721, 6.52573053999253,
            -4.81336320468984, -1.7123673353027, 6.52573053999253,
        ]).reshape((2, 4, 1))

        f_deriv = f.derivative()
        np.testing.assert_allclose(
            f_deriv(t),
            res,
        )


class TestBasisEvaluationBSpline(unittest.TestCase):
    """Test FDataBasis with B-spline basis."""

    def test_simple_evaluation(self) -> None:
        """Compare BSpline evaluation with R package fda."""
        bspline = BSplineBasis(
            domain_range=(0, 2),
            n_basis=5,
        )

        coefficients = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ])

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array([
            [  # noqa: WPS317
                1, 1.54, 1.99, 2.37, 2.7, 3,
                3.3, 3.63, 4.01, 4.46, 5,
            ],
            [  # noqa: WPS317
                6, 6.54, 6.99, 7.37, 7.7, 8,
                8.3, 8.63, 9.01, 9.46, 10,
            ],
        ])[..., np.newaxis]

        np.testing.assert_allclose(f(t), res, atol=1e-2)

    def test_evaluation_derivative(self) -> None:
        """Test the evaluation of the derivative of BSpline."""
        bspline = BSplineBasis(
            domain_range=(0, 1),
            n_basis=5,
            order=3,
        )

        coefficients = [
            [0.00078238, 0.48857741, 0.63971985, 0.23, 0.33],
            [0.01778079, 0.73440271, 0.20148638, 0.54, 0.12],
        ]

        f = FDataBasis(bspline, coefficients)

        t = np.linspace(0, 1, 4)

        f_deriv = f.derivative()
        np.testing.assert_allclose(
            f_deriv(t).round(3),
            np.array([
                [2.927, 0.453, -1.229, 0.6],
                [4.3, -1.599, 1.016, -2.52],
            ])[..., np.newaxis],
        )


class TestBasisEvaluationMonomial(unittest.TestCase):
    """Test FDataBasis with Monomial basis."""

    def test_evaluation_simple_monomial(self) -> None:
        """Compare Monomial evaluation with R package fda."""
        monomial = MonomialBasis(
            domain_range=(0, 2),
            n_basis=5,
        )

        coefficients = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ])

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 2, 11)

        # Results in R package fda
        res = np.array([
            [  # noqa: WPS317
                1.00, 1.56, 2.66, 4.79, 8.62, 15.00,
                25.00, 39.86, 61.03, 90.14, 129.00,
            ],
            [  # noqa: WPS317
                6.00, 7.81, 10.91, 16.32, 25.42, 40.00,
                62.21, 94.59, 140.08, 201.98, 284.00,
            ],
        ])[..., np.newaxis]

        np.testing.assert_allclose(f(t), res, atol=1e-2)

    def test_evaluation_derivative(self) -> None:
        """Test the evaluation of the derivative of Monomial."""
        monomial = MonomialBasis(
            domain_range=(0, 1),
            n_basis=3,
        )

        coefficients = [
            [1.0, 2.0, 3.0],
            [0.5, 1.4, 1.3],
        ]

        f = FDataBasis(monomial, coefficients)

        t = np.linspace(0, 1, 4)

        f_deriv = f.derivative()
        np.testing.assert_allclose(
            f_deriv(t).round(3),
            np.array([
                [2.0, 4.0, 6.0, 8.0],
                [1.4, 2.267, 3.133, 4.0],
            ])[..., np.newaxis],
        )


class TestBasisEvaluationVectorValued(unittest.TestCase):
    """Test basis for vector-valued functions."""

    def test_constant(self) -> None:
        """Test vector-valued constant basis."""
        basis_first = ConstantBasis()
        basis_second = ConstantBasis()

        basis = VectorValuedBasis([basis_first, basis_second])

        fd = FDataBasis(basis=basis, coefficients=[[1, 2], [3, 4]])

        self.assertEqual(fd.dim_codomain, 2)

        res = np.array([[[1, 2]], [[3, 4]]])

        np.testing.assert_allclose(fd(0), res)

    def test_monomial(self) -> None:
        """Test vector-valued monomial basis."""
        basis_first = ConstantBasis(domain_range=(0, 5))
        basis_second = MonomialBasis(
            n_basis=3,
            domain_range=(0, 5),
        )

        basis = VectorValuedBasis([basis_first, basis_second])

        fd = FDataBasis(
            basis=basis,
            coefficients=[
                [1, 2, 3, 4],
                [3, 4, 5, 6],
            ],
        )

        self.assertEqual(fd.dim_codomain, 2)

        np.testing.assert_allclose(fd.domain_range[0], (0, 5))

        res = np.array([
            [[1, 2], [1, 9], [1, 24]],
            [[3, 4], [3, 15], [3, 38]],
        ])

        np.testing.assert_allclose(fd([0, 1, 2]), res)


class TestBasisEvaluationTensor(unittest.TestCase):
    """Test tensor basis for multivariable functions."""

    def test_tensor_monomial_constant(self) -> None:
        """Test monomial ⊗ constant basis."""
        basis = TensorBasis([
            MonomialBasis(n_basis=2),
            ConstantBasis(),
        ])

        fd = FDataBasis(
            basis=basis,
            coefficients=[1, 1],
        )

        self.assertEqual(fd.dim_domain, 2)
        self.assertEqual(fd.dim_codomain, 1)

        np.testing.assert_allclose(fd([0, 0]), [[[1]]])

        np.testing.assert_allclose(fd([0.5, 0.5]), [[[1.5]]])

        np.testing.assert_allclose(
            fd([(0, 0), (0.5, 0.5)]),
            [[[1.0], [1.5]]],
        )

        fd_grid = fd.to_grid()

        fd2 = fd_grid.to_basis(basis)

        np.testing.assert_allclose(fd.coefficients, fd2.coefficients)


if __name__ == '__main__':
    unittest.main()
