"""Test for PProductMetric."""

import re

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from skfda.misc.metrics import (
    PProductMetric,
    l1_distance,
    l2_distance,
    pproduct_metric,
)
from skfda.representation.grid import FDataGrid
from skfda.typing._numpy import NDArrayFloat


@pytest.fixture
def grid() -> NDArrayFloat:
    """Return a grid of 100 evenly spaced points in [0, 1]."""
    return np.linspace(0, 1, 100)


@pytest.fixture
def fd1(grid: NDArrayFloat) -> FDataGrid:
    """Return a FDataGrid object containing a sine function."""
    data = np.sin(2 * np.pi * grid)[None, :, None]
    return FDataGrid(data, grid_points=grid)


@pytest.fixture
def fd2(grid: NDArrayFloat) -> FDataGrid:
    """Return a FDataGrid object containing a phase-shifted sine function."""
    data = np.sin(2 * np.pi * grid + 0.1)[None, :, None]
    return FDataGrid(data, grid_points=grid)


def test_ndarray_distance_basic() -> None:
    """Check Euclidean distance between two numeric DataFrames."""
    df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
    df2 = pd.DataFrame({"a": [4.0], "b": [6.0]})

    # Metrics by column name
    metrics = {"a": l2_distance, "b": l1_distance}

    # But weights must be a list or ndarray (in column order)
    weights = np.array([2.0, 2.0])

    metric: PProductMetric[pd.DataFrame, FDataGrid] = PProductMetric(
        p=2.0,
        metrics=metrics,
        weights=weights,
    )

    dist = metric(df1, df2)

    expected = np.sqrt((3.0**2 + 4.0**2) * 2)  # sqrt((9 + 16) * 2)
    assert np.isclose(dist, expected)


def test_fdgrid_distance_single_component(
    fd1: FDataGrid,
    fd2: FDataGrid,
) -> None:
    """Test distance computation between two FDataGrid objects."""
    metric: PProductMetric[FDataGrid, FDataGrid] = PProductMetric(p=2.0)
    dist = metric(fd1, fd2)
    assert isinstance(dist, float)
    assert dist >= 0


def test_fdgrid_distance_multi_component(fd1: FDataGrid) -> None:
    """Test metric computation with multivariate FDataGrid (2D codomain)."""
    fd_mult = FDataGrid(
        data_matrix=np.concatenate([fd1.data_matrix, fd1.data_matrix], axis=2),
        grid_points=fd1.grid_points,
    )
    metric: PProductMetric[FDataGrid, FDataGrid] = PProductMetric(
        p=2.0,
        metrics=[l2_distance, l2_distance],
    )
    dist = metric(fd_mult, fd_mult)

    assert np.allclose(dist, np.zeros(2))


def test_dataframe_same_structure(fd1: FDataGrid) -> None:
    """Test metric for DataFrames with the same column structure."""
    df1 = pd.DataFrame({"num": [1.0], "fd": [fd1]})
    df2 = pd.DataFrame({"num": [1.0], "fd": [fd1]})

    metric: PProductMetric[pd.DataFrame, FDataGrid] = PProductMetric(
        p=2.0,
        metrics={"num": l2_distance, "fd": l2_distance},
    )
    dist = metric(df1, df2)
    assert np.isclose(dist, 0.0)


def test_dataframe_structure_mismatch_raises(fd1: FDataGrid) -> None:  # noqa: ARG001
    """Ensure a ValueError is raised when DataFrames have mismatched keys."""
    df1 = pd.DataFrame({"a": [1.0]})
    df2 = pd.DataFrame({"b": [1.0]})
    metric: PProductMetric[pd.DataFrame, FDataGrid] = PProductMetric(
        p=2.0,
        metrics={"a": l2_distance},
    )

    with pytest.raises(
        ValueError,
        match="Columns must be the same in both DataFrames",
    ):
        metric(df1, df2)


def test_invalid_weights_ndarray_shape(fd1: FDataGrid, fd2: FDataGrid) -> None:
    """ValueError is raised when weights don't match number of components."""
    fd_mult = FDataGrid(
        data_matrix=np.concatenate([fd1.data_matrix, fd2.data_matrix], axis=2),
        grid_points=fd1.grid_points,
    )
    metric: PProductMetric[FDataGrid, FDataGrid] = PProductMetric(
        p=2.0,
        weights=np.array([1.0, 2.0, 3.0]),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of weights (3) does not match"
            " the number of dimensions (2).",
        ),
    ):
        metric(fd_mult, fd_mult)


def test_invalid_metric_type_for_array() -> None:
    """Raise TypeError if the number of metrics does not match array shape."""
    a = np.array([[1.0]])
    b = np.array([[1.0]])
    metric: PProductMetric[NDArrayFloat, NDArrayFloat] = PProductMetric(
        p=2.0,
        metrics=[l2_distance, l2_distance],
    )

    with pytest.raises(TypeError):
        metric(a, b)


def test_invalid_p_value() -> None:
    """Raise ValueError when the aggregation parameter p is less than 1."""
    with pytest.raises(
        ValueError,
        match=re.escape("p (=0.5) must be equal or greater than 1."),
    ):
        PProductMetric(p=0.5)


def test_pproduct_metric_function_interface(
    fd1: FDataGrid,
    fd2: FDataGrid,
) -> None:
    """Test functional interface `pproduct_metric` behaves as expected."""
    dist = pproduct_metric(fd1, fd2, p=2.0)
    assert isinstance(dist, float)
