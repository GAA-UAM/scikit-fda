"""Tests for Pandas integration."""
import unittest

import pandas as pd

import skfda


class TestPandas(unittest.TestCase):
    """Test basic Pandas integration."""

    def setUp(self) -> None:
        """Create FDataGrid instance."""
        self.fd = skfda.FDataGrid(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7, 9],
            ],
        )
        self.fd_basis = self.fd.to_basis(
            skfda.representation.basis.BSplineBasis(
                n_basis=5,
            ),
        )

    def test_fdatagrid_series(self) -> None:
        """Test creation of a Series with FDataGrid dtype."""
        series = pd.Series(self.fd)
        self.assertIsInstance(  # noqa: PT009
            series.dtype,
            skfda.representation.grid.FDataGridDType,
        )
        self.assertEqual(len(series), self.fd.n_samples)  # noqa: PT009
        self.assertTrue(series[0].equals(self.fd[0]))  # noqa: PT009

    def test_fdatabasis_series(self) -> None:
        """Test creation of a Series with FDataBasis dtype."""
        series = pd.Series(self.fd_basis)
        self.assertIsInstance(  # noqa: PT009
            series.dtype,
            skfda.representation.basis.FDataBasisDType,
        )
        self.assertEqual(len(series), self.fd_basis.n_samples)  # noqa: PT009
        self.assertTrue(series[0].equals(self.fd_basis[0]))  # noqa: PT009

    def test_fdatagrid_dataframe(self) -> None:
        """Test creation of a DataFrame with FDataGrid dtype."""
        df = pd.DataFrame({"function": self.fd})  # noqa: PD901
        self.assertIsInstance(  # noqa: PT009
            df["function"].dtype,
            skfda.representation.grid.FDataGridDType,
        )
        self.assertEqual(len(df["function"]), self.fd.n_samples)  # noqa: PT009
        self.assertTrue(df["function"][0].equals(self.fd[0]))  # noqa: PT009

    def test_fdatabasis_dataframe(self) -> None:
        """Test creation of a DataFrame with FDataBasis dtype."""
        df = pd.DataFrame({"function": self.fd_basis})  # noqa: PD901
        self.assertIsInstance(  # noqa: PT009
            df["function"].dtype,
            skfda.representation.basis.FDataBasisDType,
        )
        self.assertEqual(len(df["function"]), self.fd_basis.n_samples)  # noqa: PT009
        self.assertTrue(df["function"][0].equals(self.fd_basis[0]))  # noqa: PT009

    def test_take(self) -> None:
        """Test behaviour of take function."""
        self.assertTrue(self.fd.take(0).equals(self.fd[0]))  # noqa: PT009
        self.assertTrue(self.fd.take(0, axis=0).equals(self.fd[0]))  # noqa: PT009
