"""Test stats functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from skfda import FDataBasis, FDataGrid, FDataIrregular
from skfda.exploratory.stats import std
from skfda.representation.basis import (
    Basis,
    FourierBasis,
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)

# Fixtures for test_std_fdatabasis_vector_valued_basis


@pytest.fixture(params=[3, 5])
def vv_n_basis1(request: Any) -> int:
    """n_basis for 1st coordinate of vector valued basis."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def vv_basis1(vv_n_basis1: int) -> Basis:
    """1-dimensional basis to test for vector valued basis."""
    # First element of the basis is assumed to be the 1 function
    return MonomialBasis(
        n_basis=vv_n_basis1, domain_range=(0, 1),
    )


@pytest.fixture(params=[FourierBasis, MonomialBasis])
def vv_basis2(request: Any, vv_n_basis2: int = 3) -> Basis:
    """1-dimensional basis to test for vector valued basis."""
    # First element of the basis is assumed to be the 1 function
    return request.param(  # type: ignore[no-any-return]
        domain_range=(0, 1), n_basis=vv_n_basis2,
    )


# Fixtures for test_std_fdatabasis_tensor_basis

@pytest.fixture(params=[FourierBasis])
def t_basis1(request: Any, t_n_basis1: int = 3) -> Basis:
    """1-dimensional basis to test for tensor basis."""
    # First element of the basis is assumed to be the 1 function
    return request.param(  # type: ignore[no-any-return]
        domain_range=(0, 1), n_basis=t_n_basis1,
    )


@pytest.fixture(params=[MonomialBasis])
def t_basis2(request: Any, t_n_basis2: int = 5) -> Basis:
    """1-dimensional basis to test for tensor basis."""
    # First element of the basis is assumed to be the 1 function
    return request.param(  # type: ignore[no-any-return]
        domain_range=(0, 1), n_basis=t_n_basis2,
    )


# Tests

def test_std_fdatairregular_1d_to_1d() -> None:
    """Test std_fdatairregular with R to R functions."""
    fd = FDataIrregular(
        start_indices=[0, 3, 7],
        points=[0, 1, 10, 0, 1, 2, 10, 0, 1, 4, 10],
        values=[0, 0, 10, 1, 1, 6, 10, 2, 2, 9, 10],
    )
    expected_std = FDataIrregular(
        start_indices=[0],
        points=[0, 1, 10],
        values=[np.sqrt(2 / 3), np.sqrt(2 / 3), 0],
    )
    actual_std = std(fd)
    assert actual_std.equals(expected_std), actual_std


def test_std_fdatagrid_1d_to_2d() -> None:
    """Test std_fdatagrid with R to R^2 functions."""
    fd = FDataGrid(
        data_matrix=[
            [[0, 1, 2, 3, 4, 5], [0, -1, -2, -3, -4, -5]],
            [[2, 3, 4, 5, 6, 7], [-2, -3, -4, -5, -6, -7]],
        ],
        grid_points=[
            [-2, -1],
            [0, 1, 2, 3, 4, 5],
        ],
    )
    expected_std_data_matrix = np.full((1, 2, 6, 1), 1)
    np.testing.assert_allclose(
        std(fd).data_matrix,
        expected_std_data_matrix,
    )


def test_std_fdatagrid_2d_to_2d() -> None:
    """Test std_fdatagrid with R to R^2 functions."""
    fd = FDataGrid(
        data_matrix=[
            [
                [[10, 11], [10, 12], [11, 14]],
                [[15, 16], [12, 15], [20, 13]],
            ],
            [
                [[11, 12], [11, 13], [12, 13]],
                [[14, 15], [11, 16], [21, 12]],
            ],
        ],
        grid_points=[
            [0, 1],
            [0, 1, 2],
        ],
    )
    expected_std_data_matrix = np.full((1, 2, 3, 2), np.sqrt(1 / 4))
    np.testing.assert_allclose(
        std(fd).data_matrix,
        expected_std_data_matrix,
    )


def test_std_fdatabasis_vector_valued_basis(
    vv_basis1: Basis,
    vv_basis2: Basis,
) -> None:
    """Test std_fdatabasis with a vector valued basis."""
    basis = VectorValuedBasis([vv_basis1, vv_basis2])

    # coefficients of the function===(1, 1)
    one_coefficients = np.concatenate((
        np.pad([1], (0, vv_basis1.n_basis - 1)),
        np.pad([1], (0, vv_basis2.n_basis - 1)),
    ))

    fd = FDataBasis(
        basis=basis,
        coefficients=[np.zeros(basis.n_basis), one_coefficients],
    )

    np.testing.assert_allclose(
        std(fd, correction=1).coefficients,
        np.array([np.sqrt(1 / 2) * one_coefficients]),
        rtol=1e-7,
        atol=1e-7,
    )


def test_std_fdatabasis_tensor_basis(
    t_basis1: Basis,
    t_basis2: Basis,
) -> None:
    """Test std_fdatabasis with a vector valued basis."""
    basis = TensorBasis([t_basis1, t_basis2])

    # coefficients of the function===1
    one_coefficients = np.pad([1], (0, basis.n_basis - 1))

    fd = FDataBasis(
        basis=basis,
        coefficients=[np.zeros(basis.n_basis), one_coefficients],
    )

    np.testing.assert_allclose(
        std(fd, correction=1).coefficients,
        np.array([np.sqrt(1 / 2) * one_coefficients]),
        rtol=1e-7,
        atol=1e-7,
    )


def test_std_fdatabasis_2d_to_2d() -> None:
    """Test std_fdatabasis with R^2 to R^2 basis."""
    basis = VectorValuedBasis([
        TensorBasis([
            MonomialBasis(domain_range=(0, 1), n_basis=2),
            MonomialBasis(domain_range=(0, 1), n_basis=2),
        ]),
        TensorBasis([
            MonomialBasis(domain_range=(0, 1), n_basis=2),
            MonomialBasis(domain_range=(0, 1), n_basis=2),
        ]),
    ])
    fd = FDataBasis(
        basis=basis,
        coefficients=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
        ],
    )
    expected_coefficients = np.array([[np.sqrt(1 / 2), 0, 0, 0] * 2])

    np.testing.assert_allclose(
        std(fd, correction=1).coefficients,
        expected_coefficients,
        rtol=1e-7,
        atol=1e-7,
    )
