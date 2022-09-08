"""Test FDataGrid behaviour."""
import unittest

import numpy as np
import scipy.stats.mstats
from mpl_toolkits.mplot3d import axes3d

from skfda import FDataGrid, concatenate
from skfda.exploratory import stats


class TestFDataGrid(unittest.TestCase):
    """Test the FDataGrid representation."""

    def test_init(self) -> None:
        """Test creation."""
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]),
        )
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points, np.array([[0, 0.25, 0.5, 0.75, 1]]),
        )

    def test_copy_equals(self) -> None:
        """Test that copies compare equals."""
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        self.assertTrue(fd.equals(fd.copy()))

    def test_mean(self) -> None:
        """Test aritmetic mean."""
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.mean(fd)
        np.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        )
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points,
            np.array([[0, 0.25, 0.5, 0.75, 1]]),
        )

    def test_gmean(self) -> None:
        """Test geometric mean."""
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.gmean(fd)
        np.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            scipy.stats.mstats.gmean(
                np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]),
            ),
        )
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points,
            np.array([[0, 0.25, 0.5, 0.75, 1]]),
        )

    def test_slice(self) -> None:
        """Test slicing behaviour."""
        t = (5, 3)
        fd = FDataGrid(data_matrix=np.ones(t))
        fd = fd[1:3]
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1, 1, 1], [1, 1, 1]]),
        )

    def test_concatenate(self) -> None:
        """
        Test concatenation.

        Ensure that the original argument and coordinate names are kept.

        """
        fd1 = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        fd2 = FDataGrid([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])

        fd1.argument_names = ("x",)
        fd1.coordinate_names = ("y",)
        fd = fd1.concatenate(fd2)

        self.assertEqual(fd.n_samples, 4)
        self.assertEqual(fd.dim_codomain, 1)
        self.assertEqual(fd.dim_domain, 1)
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ],
        )
        self.assertEqual(
            fd1.argument_names,
            fd.argument_names,
        )
        self.assertEqual(
            fd1.coordinate_names,
            fd.coordinate_names,
        )

    def test_concatenate_coordinates(self) -> None:
        """
        Test concatenation as coordinates.

        Ensure that the coordinate names are concatenated.

        """
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd2 = FDataGrid([[3, 4, 5, 6], [4, 5, 6, 7]])

        fd1.argument_names = ("x",)
        fd1.coordinate_names = ("y",)
        fd2.argument_names = ("w",)
        fd2.coordinate_names = ("t",)
        fd = fd1.concatenate(fd2, as_coordinates=True)

        self.assertEqual(fd.n_samples, 2)
        self.assertEqual(fd.dim_codomain, 2)
        self.assertEqual(fd.dim_domain, 1)

        np.testing.assert_array_equal(
            fd.data_matrix,
            [
                [[1, 3], [2, 4], [3, 5], [4, 6]],
                [[2, 4], [3, 5], [4, 6], [5, 7]],
            ],
        )

        # Testing labels
        self.assertEqual(("y", "t"), fd.coordinate_names)
        fd2.coordinate_names = None  # type: ignore[assignment]
        fd = fd1.concatenate(fd2, as_coordinates=True)
        self.assertEqual(("y", None), fd.coordinate_names)
        fd1.coordinate_names = None  # type: ignore[assignment]
        fd = fd1.concatenate(fd2, as_coordinates=True)
        self.assertEqual((None, None), fd.coordinate_names)

    def test_concatenate_function(self) -> None:
        """Test the concatenate function (as opposed to method)."""
        sample1 = np.arange(0, 10)
        sample2 = np.arange(10, 20)
        fd1 = FDataGrid([sample1])
        fd2 = FDataGrid([sample2])

        fd1.argument_names = ("x",)
        fd1.coordinate_names = ("y",)
        fd = concatenate([fd1, fd2])

        self.assertEqual(fd.n_samples, 2)
        self.assertEqual(fd.dim_codomain, 1)
        self.assertEqual(fd.dim_domain, 1)
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            [sample1, sample2],
        )
        self.assertEqual(
            fd1.argument_names,
            fd.argument_names,
        )
        self.assertEqual(
            fd1.coordinate_names,
            fd.coordinate_names,
        )

    def test_coordinates(self) -> None:
        """Test coordinate access and iteration."""
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd1.argument_names = ("x",)
        fd1.coordinate_names = ("y",)
        fd2 = FDataGrid([[3, 4, 5, 6], [4, 5, 6, 7]])
        fd = fd1.concatenate(fd2, as_coordinates=True)

        # Indexing with number
        np.testing.assert_array_equal(
            fd.coordinates[0].data_matrix,
            fd1.data_matrix,
        )
        np.testing.assert_array_equal(
            fd.coordinates[1].data_matrix,
            fd2.data_matrix,
        )

        # Iteration
        for fd_j, fd_i in zip([fd1, fd2], fd.coordinates):
            np.testing.assert_array_equal(fd_j.data_matrix, fd_i.data_matrix)

        fd3 = fd1.concatenate(fd2, fd1, fd, as_coordinates=True)

        # Multiple indexation
        self.assertEqual(fd3.dim_codomain, 5)
        np.testing.assert_array_equal(
            fd3.coordinates[:2].data_matrix,
            fd.data_matrix,
        )
        np.testing.assert_array_equal(
            fd3.coordinates[-2:].data_matrix,
            fd.data_matrix,
        )
        index_original = np.array((False, False, True, False, True))
        np.testing.assert_array_equal(
            fd3.coordinates[index_original].data_matrix,
            fd.data_matrix,
        )

    def test_add(self) -> None:
        """Test addition with different objects."""
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])

        fd2 = fd1 + fd1
        np.testing.assert_array_equal(
            fd2.data_matrix[..., 0],  # noqa: WPS204
            [[2, 4, 6, 8], [4, 6, 8, 10]],
        )

        fd2 = fd1 + 2
        np.testing.assert_array_equal(
            fd2.data_matrix[..., 0],
            [[3, 4, 5, 6], [4, 5, 6, 7]],
        )

        fd2 = fd1 + np.array(2)
        np.testing.assert_array_equal(
            fd2.data_matrix[..., 0],
            [[3, 4, 5, 6], [4, 5, 6, 7]],
        )

        fd2 = fd1 + np.array([2])
        np.testing.assert_array_equal(
            fd2.data_matrix[..., 0],
            [[3, 4, 5, 6], [4, 5, 6, 7]],
        )

        fd2 = fd1 + np.array([1, 2])
        np.testing.assert_array_equal(
            fd2.data_matrix[..., 0],
            [[2, 3, 4, 5], [4, 5, 6, 7]],
        )

    def test_composition(self) -> None:
        """Test function composition."""
        X, Y, Z = axes3d.get_test_data(1.2)

        data_matrix = [Z.T]
        grid_points = [X[0, :], Y[:, 0]]

        g = FDataGrid(data_matrix, grid_points)
        self.assertEqual(g.dim_domain, 2)
        self.assertEqual(g.dim_codomain, 1)

        t = np.linspace(0, 2 * np.pi, 100)

        data_matrix = [10 * np.array([np.cos(t), np.sin(t)]).T]
        f = FDataGrid(data_matrix, t)
        self.assertEqual(f.dim_domain, 1)
        self.assertEqual(f.dim_codomain, 2)

        gof = g.compose(f)
        self.assertEqual(gof.dim_domain, 1)
        self.assertEqual(gof.dim_codomain, 1)


class TestEvaluateFDataGrid(unittest.TestCase):
    """Test FDataGrid evaluation."""

    def setUp(self) -> None:
        """Create bidimensional FDataGrid."""
        data_matrix = np.array(
            [
                np.tile([0, 1, 2], (2, 2, 1)),
                np.tile([3, 4, 5], (2, 2, 1)),
            ])

        grid_points = [[0, 1], [0, 1]]

        fd = FDataGrid(data_matrix, grid_points=grid_points)
        self.assertEqual(fd.n_samples, 2)
        self.assertEqual(fd.dim_domain, 2)
        self.assertEqual(fd.dim_codomain, 3)

        self.fd = fd

    def test_evaluate_aligned(self) -> None:
        """Check normal evaluation."""
        res = self.fd([(0, 0), (1, 1), (2, 2), (3, 3)])
        expected = np.array([
            np.tile([0, 1, 2], (4, 1)),
            np.tile([3, 4, 5], (4, 1)),
        ])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_unaligned(self) -> None:
        """Check unaligned evaluation."""
        res = self.fd(
            [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
                [(1, 7), (5, 2), (3, 4), (6, 1)],
            ],
            aligned=False,
        )
        expected = np.array(
            [
                np.tile([0, 1, 2], (4, 1)),
                np.tile([3, 4, 5], (4, 1)),
            ],
        )

        np.testing.assert_allclose(res, expected)

    def test_evaluate_grid_aligned(self) -> None:
        """Test evaluation in aligned grid."""
        res = self.fd([[0, 1], [1, 2]], grid=True)
        expected = np.array([
            np.tile([0, 1, 2], (2, 2, 1)),
            np.tile([3, 4, 5], (2, 2, 1)),
        ])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_grid_unaligned(self) -> None:
        """Test evaluation with a different grid per curve."""
        res = self.fd(
            [[[0, 1], [1, 2]], [[3, 4], [5, 6]]],
            grid=True,
            aligned=False,
        )
        expected = np.array([
            np.tile([0, 1, 2], (2, 2, 1)),
            np.tile([3, 4, 5], (2, 2, 1)),
        ])

        np.testing.assert_allclose(res, expected)


if __name__ == '__main__':
    unittest.main()
