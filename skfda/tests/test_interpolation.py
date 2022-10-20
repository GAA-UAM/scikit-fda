"""Tests for FDataGrid's interpolation."""
import unittest

import numpy as np

from skfda import FDataGrid
from skfda.representation.interpolation import SplineInterpolation


class TestEvaluationSplineUnivariate(unittest.TestCase):
    """Test the evaluation of univariate spline interpolation."""

    def setUp(self) -> None:
        """
        Define the data.

        The data matrix consists on functions (x**2, (9-x)**2).

        """
        self.data_matrix_1_1 = [
            np.arange(10)**2,
            np.arange(start=9, stop=-1, step=-1)**2,
        ]

        self.grid_points = np.arange(10)

    def test_evaluation_linear_simple(self) -> None:
        """Test basic usage of linear evaluation."""
        f = FDataGrid(self.data_matrix_1_1, grid_points=self.grid_points)

        # Test interpolation in nodes
        np.testing.assert_allclose(
            f(self.grid_points)[..., 0],
            self.data_matrix_1_1,
        )

        # Test evaluation in a list of times
        np.testing.assert_allclose(
            f([0.5, 1.5, 2.5]),
            np.array([
                [[0.5], [2.5], [6.5]],
                [[72.5], [56.5], [42.5]],
            ]),
        )

    def test_evaluation_linear_point(self) -> None:
        """Test the evaluation of a single point."""
        f = FDataGrid(self.data_matrix_1_1, grid_points=self.grid_points)

        # Test a single point
        np.testing.assert_allclose(
            f(5.3),
            np.array([[[28.3]], [[13.9]]]),
        )
        np.testing.assert_allclose(
            f([3]), np.array([[[9]], [[36]]]),
        )
        np.testing.assert_allclose(
            f((2,)), np.array([[[4]], [[49]]]),
        )

    def test_evaluation_linear_grid(self) -> None:
        """Test grid evaluation. With domain dimension = 1."""
        f = FDataGrid(self.data_matrix_1_1, grid_points=self.grid_points)

        # Test interpolation in nodes
        np.testing.assert_allclose(
            f(self.grid_points)[..., 0],
            self.data_matrix_1_1,
        )

        res = np.array([[[0.5], [2.5], [6.5]], [[72.5], [56.5], [42.5]]])
        t = [0.5, 1.5, 2.5]

        # Test evaluation in a list of times
        np.testing.assert_allclose(f(t, grid=True), res)
        np.testing.assert_allclose(f((t,), grid=True), res)
        np.testing.assert_allclose(f([t], grid=True), res)
        # Single point with grid
        np.testing.assert_allclose(
            f(3, grid=True),
            np.array([[[9]], [[36]]]),
        )

        # Check erroneous axis
        with self.assertRaises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_linear_unaligned(self) -> None:
        """Test unaligned evaluation."""
        f = FDataGrid(self.data_matrix_1_1, grid_points=self.grid_points)

        np.testing.assert_allclose(
            f([[1], [8]], aligned=False),
            np.array([[[1]], [[1]]]),
        )

        t = np.linspace(4, 6, 4)
        np.testing.assert_allclose(
            f([t, 9 - t], aligned=False),
            np.array([
                [[16], [22], [28.67], [36]],
                [[16], [22], [28.67], [36]],
            ]),
            rtol=1e-3,
        )

        # Same length than nsample
        t = np.linspace(4, 6, 2)
        np.testing.assert_allclose(
            f([t, 9 - t], aligned=False),
            np.array([[[16], [36]], [[16], [36]]]),
        )

    def test_evaluation_cubic_simple(self) -> None:
        """Test basic usage of cubic evaluation."""
        f = FDataGrid(
            self.data_matrix_1_1,
            grid_points=self.grid_points,
            interpolation=SplineInterpolation(3),
        )

        # Test interpolation in nodes
        np.testing.assert_allclose(
            f(self.grid_points)[..., 0],
            self.data_matrix_1_1,
            atol=1e-8,
        )

        # Test evaluation in a list of times
        np.testing.assert_allclose(
            f([0.5, 1.5, 2.5]),
            np.array([
                [[0.25], [2.25], [6.25]],
                [[72.25], [56.25], [42.25]],
            ]),
        )

    def test_evaluation_cubic_point(self) -> None:
        """Test the evaluation of a single point."""
        f = FDataGrid(
            self.data_matrix_1_1,
            grid_points=self.grid_points,
            interpolation=SplineInterpolation(3),
        )

        # Test a single point
        np.testing.assert_allclose(
            f(5.3),
            np.array([[[28.09]], [[13.69]]]),
        )

        np.testing.assert_allclose(
            f([3]),
            np.array([[[9]], [[36]]]),
        )
        np.testing.assert_allclose(
            f((2,)),
            np.array([[[4]], [[49]]]),
        )

    def test_evaluation_cubic_grid(self) -> None:
        """Test cubic grid evaluation."""
        f = FDataGrid(
            self.data_matrix_1_1,
            grid_points=self.grid_points,
            interpolation=SplineInterpolation(3),
        )

        t = [0.5, 1.5, 2.5]
        res = np.array([
            [[0.25], [2.25], [6.25]],
            [[72.25], [56.25], [42.25]],
        ])

        # Test evaluation in a list of times
        np.testing.assert_allclose(f(t, grid=True), res)
        np.testing.assert_allclose(f((t,), grid=True), res)
        np.testing.assert_allclose(f([t], grid=True), res)

        # Single point with grid
        np.testing.assert_allclose(
            f(3, grid=True),
            np.array([[[9]], [[36]]]),
        )

        # Check erroneous axis
        with self.assertRaises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_cubic_unaligned(self) -> None:
        """Test cubic unaligned evaluation."""
        f = FDataGrid(
            self.data_matrix_1_1,
            grid_points=self.grid_points,
            interpolation=SplineInterpolation(3),
        )

        np.testing.assert_allclose(
            f([[1], [8]], aligned=False),
            np.array([[[1]], [[1]]]),
        )

        t = np.linspace(4, 6, 4)
        np.testing.assert_allclose(
            f([t, 9 - t], aligned=False),
            np.array([
                [[16], [21.78], [28.44], [36]],
                [[16], [21.78], [28.44], [36]],
            ]),
            rtol=1e-3,
        )

        # Same length than nsample
        t = np.linspace(4, 6, 2)
        np.testing.assert_allclose(
            f([t, 9 - t], aligned=False),
            np.array([
                [[16], [36]], [[16], [36]],
            ]),
        )

    def test_evaluation_nodes(self) -> None:
        """Test interpolation in nodes for all dimensions."""
        for degree in range(1, 6, 2):
            interpolation = SplineInterpolation(degree)

            f = FDataGrid(
                self.data_matrix_1_1,
                grid_points=self.grid_points,
                interpolation=interpolation,
            )

            # Test interpolation in nodes
            np.testing.assert_allclose(
                f(self.grid_points)[..., 0],
                self.data_matrix_1_1,
                atol=1e-8,
            )

    def test_error_degree(self) -> None:
        """Check unsupported spline degrees."""
        with self.assertRaises(ValueError):
            interpolation = SplineInterpolation(7)
            f = FDataGrid(
                self.data_matrix_1_1,
                grid_points=self.grid_points,
                interpolation=interpolation,
            )
            f(1)

        with self.assertRaises(ValueError):
            interpolation = SplineInterpolation(0)
            f = FDataGrid(
                self.data_matrix_1_1,
                grid_points=self.grid_points,
                interpolation=interpolation,
            )
            f(1)


class TestEvaluationSplineArbitraryImage(unittest.TestCase):
    """Test spline for arbitary image dimension."""

    def setUp(self) -> None:
        """
        Define the data.

        The data matrix consists on functions (x**2, (9-x)**2).

        """
        self.t = np.arange(10)

        data_1 = np.array([
            np.arange(10)**2,
            np.arange(start=9, stop=-1, step=-1)**2,
        ])
        data_2 = np.sin(np.pi / 81 * data_1)

        self.data_matrix_1_n = np.dstack((data_1, data_2))

        self.interpolation = SplineInterpolation(interpolation_order=2)

    def test_evaluation_simple(self) -> None:
        """Test basic usage of evaluation."""
        f = FDataGrid(
            self.data_matrix_1_n,
            grid_points=np.arange(10),
            interpolation=self.interpolation,
        )

        # Test interpolation in nodes
        np.testing.assert_allclose(
            f(self.t),
            self.data_matrix_1_n,
            atol=1e-8,
        )
        # Test evaluation in a list of times
        np.testing.assert_allclose(
            f([1.5, 2.5, 3.5]),
            np.array([
                [
                    [2.25, 0.087212],
                    [6.25, 0.240202],
                    [12.25, 0.45773],
                ],
                [
                    [56.25, 0.816142],
                    [42.25, 0.997589],
                    [30.25, 0.922146],
                ],
            ]),
            rtol=1e-5,
        )

    def test_evaluation_point(self) -> None:
        """Test the evaluation of a single point."""
        f = FDataGrid(
            self.data_matrix_1_n,
            grid_points=np.arange(10),
            interpolation=self.interpolation,
        )

        # Test a single point
        np.testing.assert_allclose(
            f(5.3),
            np.array([
                [[28.09, 0.885526]],
                [[13.69, 0.50697]],
            ]),
            rtol=1e-6,
        )

    def test_evaluation_grid(self) -> None:
        """Test grid evaluation."""
        f = FDataGrid(
            self.data_matrix_1_n,
            grid_points=np.arange(10),
            interpolation=SplineInterpolation(2),
        )

        t = [1.5, 2.5, 3.5]
        res = np.array([
            [
                [2.25, 0.08721158],
                [6.25, 0.24020233],
                [12.25, 0.4577302],
            ],
            [
                [56.25, 0.81614206],
                [42.25, 0.99758925],
                [30.25, 0.92214607],
            ],
        ])

        # Test evaluation in a list of times
        np.testing.assert_allclose(f(t, grid=True), res)
        np.testing.assert_allclose(f((t,), grid=True), res)
        np.testing.assert_allclose(f([t], grid=True), res)

        # Check erroneous axis
        with self.assertRaises(ValueError):
            f((t, t), grid=True)

    def test_evaluation_unaligned(self) -> None:
        """Test unaligned evaluation."""
        f = FDataGrid(
            self.data_matrix_1_n,
            grid_points=self.t,
            interpolation=self.interpolation,
        )

        np.testing.assert_allclose(
            f([[1], [4]], aligned=False)[0],
            f(1)[0],
        )
        np.testing.assert_allclose(
            f([[1], [4]], aligned=False)[1],
            f(4)[1],
        )

    def test_evaluation_nodes(self) -> None:
        """Test interpolation in nodes for all dimensions."""
        for degree in range(1, 6, 2):
            interpolation = SplineInterpolation(degree)

            f = FDataGrid(
                self.data_matrix_1_n,
                grid_points=np.arange(10),
                interpolation=interpolation,
            )

            # Test interpolation in nodes
            np.testing.assert_allclose(
                f(np.arange(10)),
                self.data_matrix_1_n,
                atol=1e-8,
            )


@np.vectorize
def _coordinate_function(
    *args: float,
) -> np.typing.NDArray[np.float_]:
    _, *domain_indexes, _ = args
    return np.sum(domain_indexes)  # type: ignore[no-any-return]


class TestEvaluationSplineArbitraryDim(unittest.TestCase):
    """Test spline for arbitrary domain and image dimensions."""

    def test_evaluation_middle_linear(self) -> None:
        """Test linear interpolation in the middle point of a grid square."""
        dim_codomain = 4
        n_samples = 2

        for dim_domain in range(1, 6):
            grid_points = [np.array([0, 1]) for _ in range(dim_domain)]
            data_matrix = np.fromfunction(
                function=_coordinate_function,
                shape=(n_samples,) + (2,) * dim_domain + (dim_codomain,),
            )

            f = FDataGrid(data_matrix, grid_points=grid_points)

            evaluation = f([
                [0] * dim_domain,
                [0.5] * dim_domain,
                [1] * dim_domain,
            ])

            self.assertEqual(evaluation.shape, (n_samples, 3, dim_codomain))

            for i in range(n_samples):
                for j in range(dim_codomain):
                    np.testing.assert_allclose(
                        evaluation[i, ..., j],
                        [0, dim_domain * 0.5, dim_domain],
                    )


if __name__ == '__main__':
    unittest.main()
