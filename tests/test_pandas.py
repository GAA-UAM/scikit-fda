import unittest

import pandas as pd
import skfda


class TestPandas(unittest.TestCase):

    def setUp(self):
        self.fd = skfda.FDataGrid(
            [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 9]])
        self.fd_basis = self.fd.to_basis(skfda.representation.basis.BSpline(
            domain_range=self.fd.domain_range, nbasis=5))

    def test_fdatagrid_series(self):
        series = pd.Series(self.fd)
        self.assertEqual(
            series.dtype, skfda.representation.grid.FDataGridDType)
        self.assertEqual(len(series), self.fd.nsamples)
        self.assertEqual(series[0], self.fd[0])

    def test_fdatabasis_series(self):
        series = pd.Series(self.fd_basis)
        self.assertEqual(
            series.dtype, skfda.representation.basis.FDataBasisDType)
        self.assertEqual(len(series), self.fd_basis.nsamples)
        self.assertEqual(series[0], self.fd_basis[0])

    def test_fdatagrid_dataframe(self):
        df = pd.DataFrame({"function": self.fd})
        self.assertEqual(
            df["function"].dtype, skfda.representation.grid.FDataGridDType)
        self.assertEqual(len(df["function"]), self.fd.nsamples)
        self.assertEqual(df["function"][0], self.fd[0])

    def test_fdatabasis_dataframe(self):
        df = pd.DataFrame({"function": self.fd_basis})
        self.assertEqual(
            df["function"].dtype, skfda.representation.basis.FDataBasisDType)
        self.assertEqual(len(df["function"]), self.fd_basis.nsamples)
        self.assertEqual(df["function"][0], self.fd_basis[0])
