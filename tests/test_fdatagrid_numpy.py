from skfda import FDataGrid
import unittest
import numpy as np


class TestFDataGridNumpy(unittest.TestCase):

    def test_monary_ufunc(self):
        data_matrix = np.arange(15).reshape(3, 5)

        fd = FDataGrid(data_matrix)

        fd_sqrt = np.sqrt(fd)

        fd_sqrt_build = FDataGrid(np.sqrt(data_matrix))

        self.assertTrue(fd_sqrt.equals(fd_sqrt_build))

    def test_binary_ufunc(self):
        data_matrix = np.arange(15).reshape(3, 5)
        data_matrix2 = 2 * np.arange(15).reshape(3, 5)

        fd = FDataGrid(data_matrix)
        fd2 = FDataGrid(data_matrix2)

        fd_mul = np.multiply(fd, fd2)

        fd_mul_build = FDataGrid(data_matrix * data_matrix2)

        self.assertTrue(fd_mul.equals(fd_mul_build))

    def test_out_ufunc(self):
        data_matrix = np.arange(15.).reshape(3, 5)
        data_matrix_copy = np.copy(data_matrix)

        fd = FDataGrid(data_matrix)

        np.sqrt(fd, out=fd)

        fd_sqrt_build = FDataGrid(np.sqrt(data_matrix_copy))

        self.assertTrue(fd.equals(fd_sqrt_build))


if __name__ == '__main__':
    print()
    unittest.main()
