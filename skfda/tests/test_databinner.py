"""Tests for DataBinner module."""

import re
from typing import Any, Optional, Tuple, Union

import numpy as np
import pytest

from skfda.preprocessing.binning import DataBinner
from skfda.representation import FData, FDataGrid, FDataIrregular

##############################################################################
# Example FDataGrid to check parameters
##############################################################################

fd = FDataGrid(
    grid_points=[[0, 1], [0, 1]],
    data_matrix=np.array(
        [
            [
                [1.0, 2.0],
                [4.0, 3.0],
            ],
            [
                [1.0, 2.0],
                [4.0, 3.0],
            ],
        ],
    ),
)

##############################################################################
# Fixtures
##############################################################################


@pytest.fixture(
    params=[
        [
            (3, 2),
            (0, 2),
            (np.array([0, 1, 2]), np.array([0, 1, 2])),
            "median",
            "For 2-dimensional domain, range must be a tuple with 2 " "tuples",
        ],
    ],
)
def databinner_raises_fixture(request: Any) -> list[
    Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]],
    Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
    Union[str, np.ndarray, Tuple[np.ndarray, ...]],
    str,
    str,
]:
    """Fixture for getting a DataBinner object that raises a ValueError."""
    return request.param


@pytest.fixture(
    params=[
        [
            DataBinner(bins=(1, 1)),
            fd.domain_range,
        ],
        [
            DataBinner(bins=(1, 1), range=((0, 2), (0, 2))),
            ((0, 2), (0, 2)),
        ],
        [
            DataBinner(bins=(np.array([-1, 1]), np.array([-1, 1]))),
            ((-1, 1), (-1, 1)),
        ],
        [
            DataBinner(
                bins=(np.array([-1, 1]), np.array([-1, 1])),
                range=((0, 2), (0, 2)),
            ),
            ((-1, 1), (-1, 1)),
        ],
    ],
)
def range_interactions_fixture(request: Any) -> list[FData, DataBinner, tuple]:
    """Fixture for checking domain behaviour."""
    return fd, *request.param


@pytest.fixture(
    params=[
        [
            DataBinner(
                bins=(2, 2),
                output_grid=(np.array([0, 1]), np.array([0, 1])),
            ),
            (np.array([0, 1]), np.array([0, 1])),
        ],
        [
            DataBinner(bins=(2, 2), output_grid="left"),
            (np.array([0, 0.5]), np.array([0, 0.5])),
        ],
        [
            DataBinner(bins=(2, 2), output_grid="right"),
            (np.array([0.5, 1]), np.array([0.5, 1])),
        ],
        [
            DataBinner(bins=(2, 2), output_grid="middle"),
            (np.array([0.25, 0.75]), np.array([0.25, 0.75])),
        ],
    ],
)
def output_grid_interactions_fixture(
    request: Any,
) -> list[FData, DataBinner, tuple]:
    """Fixture for studying behavior of output_grid."""
    return fd, *request.param


@pytest.fixture(
    params=[
        [
            DataBinner(bins=2),
            "Input FData must have 1 domain dimensions.",
        ],
        [
            DataBinner(
                bins=(2, 2),
                output_grid=(np.array([0.6, 1]), np.array([0, 1])),
            ),
            "Some output grid points in dimension 0 are outside their bin "
            "ranges. Ensure all values lie within [0.0, 1.0] and their "
            "intended bin.",
        ],
    ],
)
def fitting_raises_fixture(request: Any) -> list[FDataGrid, DataBinner, str]:
    """Fixture for checking mismatch between binner and FData."""
    return fd, *request.param


@pytest.fixture(
    params=[
        [
            FDataGrid(
                grid_points=np.linspace(0, 3, 4),
                data_matrix=np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [4.0, 3.0, 2.0, 1.0],
                    ],
                ),
            ),
            FDataGrid(
                grid_points=np.array([0.75, 2.25]),
                data_matrix=np.array(
                    [
                        [1.5, 3.5],
                        [3.5, 1.5],
                    ],
                ),
            ),
        ],
    ],
)
def precalc_example_data_1_1(
    request: Any,
) -> list[FData, FDataGrid]:
    """
    Fixture for getting FData objects and their expected binned results.

    Calculations for these FDataGrid, FDataIrregular objects have been done
    previously and the expected result is returned as well.
    """
    return request.param


##############################################################################
# Tests
##############################################################################


def test_raises(
    databinner_raises_fixture: list[
        Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]],
        Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
        Union[str, np.ndarray, Tuple[np.ndarray, ...]],
        str,
        str,
    ],
) -> None:
    """
    Check raises ValueError.

    Check that DataBinners raise a ValueError exception.
    """
    bins, range_, output_grid, bin_aggregation, error_msg = (
        databinner_raises_fixture
    )

    with pytest.raises(ValueError, match=error_msg):
        DataBinner(
            bins=bins,
            range=range_,
            output_grid=output_grid,
            bin_aggregation=bin_aggregation,
        )


def test_domain_range_interactions(
    range_interactions_fixture: list[FData, DataBinner, tuple],
) -> None:
    """
    Check the domain parameter behaviour.

    Check that the domain is correctly calculated for different combinations
    of range, bins and domain.
    """
    fd, binner, expected_domain = range_interactions_fixture
    binned_fd = binner.fit_transform(fd)

    np.testing.assert_array_equal(
        binned_fd.domain_range,
        expected_domain,
    )


def test_output_grid_interactions(
    output_grid_interactions_fixture: list[FData, DataBinner, tuple],
) -> None:
    """
    Check the output grid parameter behaviour.

    Check that the output grid is correctly calculated for different
    output_grid parameters.
    """
    fd, binner, expected_grid = output_grid_interactions_fixture
    binned_fd = binner.fit_transform(fd)

    np.testing.assert_array_equal(
        binned_fd.grid_points,
        expected_grid,
    )


def test_fitting_raises(
    fitting_raises_fixture: list[FDataGrid, DataBinner, str],
) -> None:
    """
    Check the domain parameter behaviour in fitting.

    Check that fit() raises a ValueError exception.
    """
    fd, binner, error_msg = fitting_raises_fixture

    with pytest.raises(ValueError, match=re.escape(error_msg)):
        binner.fit(fd)


def test_precalc_example(
    precalc_example_data_1_1: list[FData, FDataGrid],
) -> None:
    """
    Check the precalculated example for binned FData.

    Compare the theoretical precalculated results against the binned data
    with DataBinner implementation, for different FData.
    """
    fd, precalc_result = precalc_example_data_1_1
    binner = DataBinner(bins=2)
    computed_result = binner.fit_transform(fd)

    np.testing.assert_array_equal(
        precalc_result.data_matrix,
        computed_result.data_matrix,
    )

    np.testing.assert_array_equal(
        precalc_result.grid_points,
        computed_result.grid_points,
    )


    # missing tests for all parameter inputs
    # missing irregular for 1->1
    # missing tests for results in combinations of domain-codomain