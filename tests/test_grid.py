from skfda import FDataGrid, concatenate
from skfda.exploratory import stats
import unittest

from mpl_toolkits.mplot3d import axes3d
import scipy.stats.mstats

import numpy as np


class TestFDataGrid(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_init(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]))
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points, np.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_copy_equals(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        self.assertTrue(fd.equals(fd.copy()))

    def test_mean(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.mean(fd)
        np.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]))
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points,
            np.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_gmean(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.gmean(fd)
        np.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            scipy.stats.mstats.gmean(
                np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])))
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.grid_points,
            np.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_slice(self):
        t = (5, 3)
        fd = FDataGrid(data_matrix=np.ones(t))
        fd = fd[1:3]
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1, 1, 1], [1, 1, 1]]))

    def test_concatenate(self):
        fd1 = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        fd2 = FDataGrid([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])

        fd1.argument_names = ["x"]
        fd1.coordinate_names = ["y"]
        fd = fd1.concatenate(fd2)

        np.testing.assert_equal(fd.n_samples, 4)
        np.testing.assert_equal(fd.dim_codomain, 1)
        np.testing.assert_equal(fd.dim_domain, 1)
        np.testing.assert_array_equal(fd.data_matrix[..., 0],
                                      [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
                                       [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(fd1.argument_names, fd.argument_names)
        np.testing.assert_array_equal(
            fd1.coordinate_names, fd.coordinate_names)

    def test_concatenate_coordinates(self):
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd2 = FDataGrid([[3, 4, 5, 6], [4, 5, 6, 7]])

        fd1.argument_names = ["x"]
        fd1.coordinate_names = ["y"]
        fd2.argument_names = ["w"]
        fd2.coordinate_names = ["t"]
        fd = fd1.concatenate(fd2, as_coordinates=True)

        np.testing.assert_equal(fd.n_samples, 2)
        np.testing.assert_equal(fd.dim_codomain, 2)
        np.testing.assert_equal(fd.dim_domain, 1)

        np.testing.assert_array_equal(fd.data_matrix,
                                      [[[1, 3], [2, 4], [3, 5], [4, 6]],
                                       [[2, 4], [3, 5], [4, 6], [5, 7]]])

        # Testing labels
        np.testing.assert_array_equal(("y", "t"), fd.coordinate_names)
        fd2.coordinate_names = None
        fd = fd1.concatenate(fd2, as_coordinates=True)
        np.testing.assert_array_equal(("y", None), fd.coordinate_names)
        fd1.coordinate_names = None
        fd = fd1.concatenate(fd2, as_coordinates=True)
        np.testing.assert_equal((None, None), fd.coordinate_names)

    def test_concatenate2(self):
        sample1 = np.arange(0, 10)
        sample2 = np.arange(10, 20)
        fd1 = FDataGrid([sample1])
        fd2 = FDataGrid([sample2])

        fd1.argument_names = ["x"]
        fd1.coordinate_names = ["y"]
        fd = concatenate([fd1, fd2])

        np.testing.assert_equal(fd.n_samples, 2)
        np.testing.assert_equal(fd.dim_codomain, 1)
        np.testing.assert_equal(fd.dim_domain, 1)
        np.testing.assert_array_equal(fd.data_matrix[..., 0], [sample1,
                                                               sample2])
        np.testing.assert_array_equal(fd1.argument_names, fd.argument_names)
        np.testing.assert_array_equal(
            fd1.coordinate_names, fd.coordinate_names)

    def test_coordinates(self):
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd1.argument_names = ["x"]
        fd1.coordinate_names = ["y"]
        fd2 = FDataGrid([[3, 4, 5, 6], [4, 5, 6, 7]])
        fd = fd1.concatenate(fd2, as_coordinates=True)

        # Indexing with number
        np.testing.assert_array_equal(fd.coordinates[0].data_matrix,
                                      fd1.data_matrix)
        np.testing.assert_array_equal(fd.coordinates[1].data_matrix,
                                      fd2.data_matrix)

        # Iteration
        for fd_j, fd_i in zip([fd1, fd2], fd.coordinates):
            np.testing.assert_array_equal(fd_j.data_matrix, fd_i.data_matrix)

        fd3 = fd1.concatenate(fd2, fd1, fd, as_coordinates=True)

        # Multiple indexation
        np.testing.assert_equal(fd3.dim_codomain, 5)
        np.testing.assert_array_equal(fd3.coordinates[:2].data_matrix,
                                      fd.data_matrix)
        np.testing.assert_array_equal(fd3.coordinates[-2:].data_matrix,
                                      fd.data_matrix)
        np.testing.assert_array_equal(
            fd3.coordinates[np.array(
                (False, False, True, False, True))].data_matrix,
            fd.data_matrix)

    def test_add(self):
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])

        fd2 = fd1 + fd1
        np.testing.assert_array_equal(fd2.data_matrix[..., 0],
                                      [[2, 4, 6, 8], [4, 6, 8, 10]])

        fd2 = fd1 + 2
        np.testing.assert_array_equal(fd2.data_matrix[..., 0],
                                      [[3, 4, 5, 6], [4, 5, 6, 7]])

        fd2 = fd1 + np.array(2)
        np.testing.assert_array_equal(fd2.data_matrix[..., 0],
                                      [[3, 4, 5, 6], [4, 5, 6, 7]])

        fd2 = fd1 + np.array([2])
        np.testing.assert_array_equal(fd2.data_matrix[..., 0],
                                      [[3, 4, 5, 6], [4, 5, 6, 7]])

        fd2 = fd1 + np.array([1, 2])
        np.testing.assert_array_equal(fd2.data_matrix[..., 0],
                                      [[2, 3, 4, 5], [4, 5, 6, 7]])

    def test_composition(self):
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

    def setUp(self):
        data_matrix = np.array(
            [
                [
                    [[0, 1, 2], [0, 1, 2]],
                    [[0, 1, 2], [0, 1, 2]]
                ],
                [
                    [[3, 4, 5], [3, 4, 5]],
                    [[3, 4, 5], [3, 4, 5]]
                ]
            ])

        grid_points = [[0, 1], [0, 1]]

        fd = FDataGrid(data_matrix, grid_points=grid_points)
        self.assertEqual(fd.n_samples, 2)
        self.assertEqual(fd.dim_domain, 2)
        self.assertEqual(fd.dim_codomain, 3)

        self.fd = fd

    def test_evaluate_aligned(self):

        res = self.fd([(0, 0), (1, 1), (2, 2), (3, 3)])
        expected = np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
                             [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_unaligned(self):

        res = self.fd([[(0, 0), (1, 1), (2, 2), (3, 3)],
                       [(1, 7), (5, 2), (3, 4), (6, 1)]],
                      aligned=False)
        expected = np.array([[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
                             [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_unaligned_ragged(self):

        res = self.fd([[(0, 0), (1, 1), (2, 2), (3, 3)],
                       [(1, 7), (5, 2), (3, 4)]],
                      aligned=False)
        expected = ([[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
                     [[3, 4, 5], [3, 4, 5], [3, 4, 5]]])

        self.assertEqual(len(res), self.fd.n_samples)

        for r, e in zip(res, expected):
            np.testing.assert_allclose(r, e)

    def test_evaluate_grid_aligned(self):

        res = self.fd([[0, 1], [1, 2]], grid=True)
        expected = np.array([[[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]],
                             [[[3, 4, 5], [3, 4, 5]], [[3, 4, 5], [3, 4, 5]]]])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_grid_unaligned(self):

        res = self.fd([[[0, 1], [1, 2]], [[3, 4], [5, 6]]],
                      grid=True, aligned=False)
        expected = np.array([[[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]],
                             [[[3, 4, 5], [3, 4, 5]], [[3, 4, 5], [3, 4, 5]]]])

        np.testing.assert_allclose(res, expected)

    def test_evaluate_grid_unaligned_ragged(self):

        res = self.fd([[[0, 1], [1, 2]], [[3, 4], [5]]],
                      grid=True, aligned=False)
        expected = ([[[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]],
                     [[[3, 4, 5]], [[3, 4, 5]]]])

        for r, e in zip(res, expected):
            np.testing.assert_allclose(r, e)


if __name__ == '__main__':
    print()
    unittest.main()
