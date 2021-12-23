"""Tests of compatibility between numpy ufuncs and FDataGrid."""

import unittest
from typing import Any, Callable, TypeVar

import numpy as np
import pytest

import skfda
from skfda import FDataGrid


@pytest.fixture(params=[
    np.sqrt,
    np.absolute,
    np.round,
    np.exp,
    np.log,
    np.log10,
    np.log2,
])
def monary(request: Any) -> Any:
    """
    Fixture providing the monary function to validate.

    Not all of them are ufuncs.

    """
    return request.param


T = TypeVar("T", np.ndarray, FDataGrid)


def test_monary_ufuncs(monary: Callable[[T], T]) -> None:
    """Test that unary ufuncs can be applied to FDataGrid."""
    data_matrix = np.arange(15).reshape(3, 5) + 1

    fd = FDataGrid(data_matrix)

    fd_monary = monary(fd)

    fd_monary_build = FDataGrid(monary(data_matrix))

    assert fd_monary.equals(fd_monary_build)


def test_binary_ufunc() -> None:
    """Test that binary ufuncs can be applied to FDataGrid."""
    data_matrix = np.arange(15).reshape(3, 5)
    data_matrix2 = 2 * np.arange(15).reshape(3, 5)

    fd = FDataGrid(data_matrix)
    fd2 = FDataGrid(data_matrix2)

    fd_mul = np.multiply(fd, fd2)

    fd_mul_build = FDataGrid(data_matrix * data_matrix2)

    assert fd_mul.equals(fd_mul_build)


def test_out_ufunc(monary: Callable[..., Any]) -> None:
    """Test that the out parameter of ufuncs work for FDataGrid."""
    data_matrix = np.arange(15).reshape(3, 5) + 1
    data_matrix_copy = np.copy(data_matrix)

    fd = FDataGrid(data_matrix)

    monary(fd, out=fd)

    fd_monary_build = FDataGrid(monary(data_matrix_copy))

    assert fd.equals(fd_monary_build)


class TestOperators(unittest.TestCase):
    """Tests for operators."""

    def test_commutativity(self) -> None:
        """Test that operations with numpy arrays commute."""
        X = FDataGrid([[1, 2, 3], [4, 5, 6]])
        arr = np.array([1, 2])

        self.assertTrue((arr + X).equals((X + arr)))

    def test_commutativity_basis(self) -> None:
        """Test that operations with numpy arrays for basis commute."""
        X = FDataGrid([[1, 2, 3], [4, 5, 6]])
        arr = np.array([1, 2])
        basis = skfda.representation.basis.Fourier(n_basis=5)

        X_basis = X.to_basis(basis)

        self.assertTrue((arr * X_basis).equals((X_basis * arr)))
