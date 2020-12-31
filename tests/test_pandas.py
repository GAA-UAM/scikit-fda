import skfda
import unittest

import pandas as pd


class TestPandas(unittest.TestCase):

    def setUp(self):
        self.fd = skfda.FDataGrid(
            [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 9]])
        self.fd_basis = self.fd.to_basis(skfda.representation.basis.BSpline(
            n_basis=5))

    def test_fdatagrid_series(self):
        series = pd.Series(self.fd)
        self.assertIsInstance(
            series.dtype, skfda.representation.grid.FDataGridDType)
        self.assertEqual(len(series), self.fd.n_samples)
        self.assertTrue(series[0].equals(self.fd[0]))

    def test_fdatabasis_series(self):
        series = pd.Series(self.fd_basis)
        self.assertIsInstance(
            series.dtype, skfda.representation.basis.FDataBasisDType)
        self.assertEqual(len(series), self.fd_basis.n_samples)
        self.assertTrue(series[0].equals(self.fd_basis[0]))

    def test_fdatagrid_dataframe(self):
        df = pd.DataFrame({"function": self.fd})
        self.assertIsInstance(
            df["function"].dtype, skfda.representation.grid.FDataGridDType)
        self.assertEqual(len(df["function"]), self.fd.n_samples)
        self.assertTrue(df["function"][0].equals(self.fd[0]))

    def test_fdatabasis_dataframe(self):
        df = pd.DataFrame({"function": self.fd_basis})
        self.assertIsInstance(
            df["function"].dtype, skfda.representation.basis.FDataBasisDType)
        self.assertEqual(len(df["function"]), self.fd_basis.n_samples)
        self.assertTrue(df["function"][0].equals(self.fd_basis[0]))

    def test_take(self):
        self.assertTrue(self.fd.take(0).equals(self.fd[0]))
        self.assertTrue(self.fd.take(0, axis=0).equals(self.fd[0]))
