import numpy as np
import pandas as pd
import pytest

from skfda.representation.grid import FDataGrid
from skfda.misc.metrics import l2_distance
from skfda.ml.metrics import PProductMetric, pproduct_metric


@pytest.fixture
def grid():
    return np.linspace(0, 1, 100)


@pytest.fixture
def fd1(grid):
    data = np.sin(2 * np.pi * grid)[None, :, None]
    return FDataGrid(data, grid_points=grid)


@pytest.fixture
def fd2(grid):
    data = np.sin(2 * np.pi * grid + 0.1)[None, :, None]
    return FDataGrid(data, grid_points=grid)


def test_ndarray_distance_basic():
    a = np.array([[1.0, 2.0]])
    b = np.array([[4.0, 6.0]])
    metric = PProductMetric(p=2.0, metrics=l2_distance, weights=2.0)

    dist = metric(a, b)
    expected = np.linalg.norm(a - b) * np.sqrt(2)
    assert np.isclose(dist, expected)


def test_fdgrid_distance_single_component(fd1, fd2):
    metric = PProductMetric(p=2.0)
    dist = metric(fd1, fd2)
    assert isinstance(dist, float)
    assert dist >= 0


def test_fdgrid_distance_multi_component(fd1):
    fd_mult = FDataGrid(
        data_matrix=np.concatenate([fd1.data_matrix, fd1.data_matrix], axis=2),
        grid_points=fd1.grid_points,
    )
    metric = PProductMetric(p=2.0, metrics=[l2_distance, l2_distance])
    dist = metric(fd_mult, fd_mult)
    assert np.isclose(dist, 0.0)


def test_dataframe_same_structure(fd1):
    df1 = pd.DataFrame({
        "num": [1.0],
        "fd": [fd1],
    })
    df2 = pd.DataFrame({
        "num": [1.0],
        "fd": [fd1],
    })
    metric = PProductMetric(p=2.0, metrics={"num": l2_distance, "fd": l2_distance})
    dist = metric(df1, df2)
    assert np.isclose(dist, 0.0)


def test_dataframe_structure_mismatch_raises(fd1):
    df1 = pd.DataFrame({"a": [1.0]})
    df2 = pd.DataFrame({"b": [1.0]})
    metric = PProductMetric(p=2.0, metrics={"a": l2_distance})

    with pytest.raises(ValueError):
        metric(df1, df2)


def test_invalid_weights_ndarray_shape(fd1, fd2):
    fd_mult = FDataGrid(
        data_matrix=np.concatenate([fd1.data_matrix, fd2.data_matrix], axis=2),
        grid_points=fd1.grid_points,
    )
    metric = PProductMetric(p=2.0, weights=np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        metric(fd_mult, fd_mult)


def test_invalid_metric_type_for_array():
    a = np.array([[1.0]])
    b = np.array([[1.0]])
    metric = PProductMetric(p=2.0, metrics=[l2_distance, l2_distance])

    with pytest.raises(TypeError):
        metric(a, b)


def test_invalid_p_value():
    with pytest.raises(ValueError):
        PProductMetric(p=0.5)


def test_pproduct_metric_function_interface(fd1, fd2):
    dist = pproduct_metric(fd1, fd2, p=2.0)
    assert isinstance(dist, float)
