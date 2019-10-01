import unittest

import scipy.stats.mstats

import numpy as np
from skfda import FDataGrid
from skfda.exploratory import stats


class TestFDataGrid(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_init(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]))
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.sample_points, np.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_copy_equals(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        self.assertEqual(fd, fd.copy())

    def test_mean(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.mean(fd)
        np.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]))
        np.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        np.testing.assert_array_equal(
            fd.sample_points,
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
            fd.sample_points,
            np.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_slice(self):
        t = 10
        fd = FDataGrid(data_matrix=np.ones(t))
        fd = fd[:, 0]
        np.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            np.array([[1]]))
        np.testing.assert_array_equal(
            fd.sample_points,
            np.array([[0]]))

    def test_concatenate(self):
        fd1 = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        fd2 = FDataGrid([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])

        fd1.axes_labels = ["x", "y"]
        fd = fd1.concatenate(fd2)

        np.testing.assert_equal(fd.n_samples, 4)
        np.testing.assert_equal(fd.dim_codomain, 1)
        np.testing.assert_equal(fd.dim_domain, 1)
        np.testing.assert_array_equal(fd.data_matrix[..., 0],
                                      [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
                                       [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(fd1.axes_labels, fd.axes_labels)

    def test_concatenate_coordinates(self):
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd2 = FDataGrid([[3, 4, 5, 6], [4, 5, 6, 7]])

        fd1.axes_labels = ["x", "y"]
        fd2.axes_labels = ["w", "t"]
        fd = fd1.concatenate(fd2, as_coordinates=True)

        np.testing.assert_equal(fd.n_samples, 2)
        np.testing.assert_equal(fd.dim_codomain, 2)
        np.testing.assert_equal(fd.dim_domain, 1)

        np.testing.assert_array_equal(fd.data_matrix,
                                      [[[1, 3], [2, 4], [3, 5], [4, 6]],
                                       [[2, 4], [3, 5], [4, 6], [5, 7]]])

        # Testing labels
        np.testing.assert_array_equal(["x", "y", "t"], fd.axes_labels)
        fd1.axes_labels = ["x", "y"]
        fd2.axes_labels = None
        fd = fd1.concatenate(fd2, as_coordinates=True)
        np.testing.assert_array_equal(["x", "y", None], fd.axes_labels)
        fd1.axes_labels = None
        fd = fd1.concatenate(fd2, as_coordinates=True)
        np.testing.assert_equal(None, fd.axes_labels)

    def test_coordinates(self):
        fd1 = FDataGrid([[1, 2, 3, 4], [2, 3, 4, 5]])
        fd1.axes_labels = ["x", "y"]
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
            fd3.coordinates[(False, False, True, False, True)].data_matrix,
            fd.data_matrix)


if __name__ == '__main__':
    print()
    unittest.main()
