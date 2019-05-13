import unittest

import numpy
import scipy.stats.mstats

from skfda.exploratory import stats
from skfda import FDataGrid


class TestFDataGrid(unittest.TestCase):

    # def setUp(self): could be defined for set up before any test

    def test_init(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        numpy.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            numpy.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]))
        numpy.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        numpy.testing.assert_array_equal(
            fd.sample_points, numpy.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_mean(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.mean(fd)
        numpy.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            numpy.array([1.5, 2.5, 3.5, 4.5, 5.5]))
        numpy.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        numpy.testing.assert_array_equal(
            fd.sample_points,
            numpy.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_gmean(self):
        fd = FDataGrid([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        mean = stats.gmean(fd)
        numpy.testing.assert_array_equal(
            mean.data_matrix[0, ..., 0],
            scipy.stats.mstats.gmean(
                numpy.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])))
        numpy.testing.assert_array_equal(fd.sample_range, [(0, 1)])
        numpy.testing.assert_array_equal(
            fd.sample_points,
            numpy.array([[0., 0.25, 0.5, 0.75, 1.]]))

    def test_slice(self):
        t = 10
        fd = FDataGrid(data_matrix=numpy.ones(t))
        fd = fd[:, 0]
        numpy.testing.assert_array_equal(
            fd.data_matrix[..., 0],
            numpy.array([[1]]))
        numpy.testing.assert_array_equal(
            fd.sample_points,
            numpy.array([[0]]))


if __name__ == '__main__':
    print()
    unittest.main()
