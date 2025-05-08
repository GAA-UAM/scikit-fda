"""Tests for the centering and scaling functionality."""
from collections.abc import Generator

import numpy as np
import pytest

from skfda import FDataGrid
from skfda.exploratory.stats import (
    grand_mean,
    individual_observation_mean,
    root_integrated_sample_variance,
)
from skfda.representation._functional_data import FData
from skfda.typing._numpy import NDArrayFloat


@pytest.fixture
def sample_fdgrid() -> Generator[FDataGrid, None, None]:
    """Fixture: sample FDataGrid with 3 linear functions on [0, 1]."""
    data = np.array([
        np.linspace(0, 1, 5),        # f1(t) = t
        np.linspace(1, 2, 5),        # f2(t) = t + 1
        np.linspace(2, 3, 5),        # f3(t) = t + 2
    ])
    grid_points = np.linspace(0, 1, 5)
    return FDataGrid(data_matrix=data, grid_points=grid_points)


def test_individual_observation_mean(sample_fdgrid: FDataGrid) -> None:
    """Test individual means match row-wise average over grid points."""
    means: NDArrayFloat = individual_observation_mean(sample_fdgrid)
    expected: NDArrayFloat = np.array([[0.5], [1.5], [2.5]])
    np.testing.assert_allclose(means, expected, rtol=1e-5)


def test_grand_mean(sample_fdgrid: FDataGrid) -> None:
    """Test grand mean matches average of all individual means."""
    gm: NDArrayFloat = grand_mean(sample_fdgrid)
    np.testing.assert_allclose(gm, 1.5, rtol=1e-5)


def test_root_integrated_sample_variance(sample_fdgrid: FDataGrid) -> None:
    """Test RISV matches implementation logic."""
    risv: NDArrayFloat = root_integrated_sample_variance(sample_fdgrid)
    np.testing.assert_allclose(risv, 1, rtol=1e-5)


def test_unsupported_type() -> None:
    """Test RISV raises TypeError on invalid input."""
    with pytest.raises(TypeError):
        root_integrated_sample_variance(
            "not an FData object", # type: ignore[arg-type]
            )
