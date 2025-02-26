"""Tests for DataBinner module."""

import re
from typing import Any, Tuple, TypeAlias, Union

import numpy as np
import pytest

from skfda.preprocessing.binning import DataBinner
from skfda.representation import FData, FDataGrid, FDataIrregular

BinnerFixture: TypeAlias = list[
    Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]],
    Union[Tuple[float, float], Tuple[Tuple[float, float], ...]],
    Union[str, np.ndarray, Tuple[np.ndarray, ...]],
    str,
    str,
]

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
            "middle",
            "median",
            "For 2-dimensional domain, range must be a tuple with 2 " "tuples",
        ],
        [
            (2, 2),
            ((0, 2), (0, 2)),
            np.array([0, 1]),
            "median",
            "Output grid must be 'left', 'middle', 'right' or a 2 tuple of "
            "numpy arrays for 2-dimensional domain.",
        ],
        [
            (2, 2),
            ((0, 2), (0, 2)),
            (np.array([0, 1, 2]), np.array([0, 1, 2])),
            "median",
            "Output grid at dimension 0 has length 3, but expected 2 based "
            "on the number of bins.",
        ],
        [
            (2, 2),
            ((0, 2), (0, 2)),
            (np.array([0, 1, 1]), np.array([0, 1, 2])),
            "median",
            "Each output grid must be strictly increasing.",
        ],
    ],
)
def databinner_raises_fixture(request: Any) -> BinnerFixture:
    """Fixture for getting a DataBinner object that raises a ValueError."""
    return request.param


@pytest.fixture(
    params=[
        [
            DataBinner(bins=(1, 1)),
            fd.domain_range,
        ],
        [
            DataBinner(bins=(1, 1), domain_range=((0, 2), (0, 2))),
            ((0, 2), (0, 2)),
        ],
        [
            DataBinner(bins=(np.array([-1, 1]), np.array([-1, 1]))),
            ((-1, 1), (-1, 1)),
        ],
        [
            DataBinner(
                bins=(np.array([-1, 1]), np.array([-1, 1])),
                domain_range=((0, 2), (0, 2)),
            ),
            ((-1, 1), (-1, 1)),
        ],
    ],
)
def domain_range_interactions_fixture(
    request: Any,
) -> list[FData, DataBinner, tuple]:
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
        [
            FDataIrregular(
                start_indices=[0, 3],
                points=[[0.0], [1.0], [4.0], [1.0], [3.0], [6.0]],
                values=[[1.0], [2.0], [1.0], [3.0], [4.0], [5.0]],
            ),
            FDataGrid(
                grid_points=np.array([1.5, 4.5]),
                data_matrix=np.array(
                    [
                        [1.5, 1.0],
                        [3.0, 4.5],
                    ],
                ),
            ),
        ],
        [
            FDataGrid(
                grid_points=np.linspace(0, 3, 4),
                data_matrix=np.array(
                    [
                        [
                            [1.0, 5.0, 9.0],
                            [2.0, 6.0, 10.0],
                            [3.0, 7.0, 0.0],
                            [4.0, 8.0, 1.0],
                        ],
                        [
                            [11.0, 4.0, 3.0],
                            [12.0, 8.0, 0.0],
                            [13.0, 1.0, 0.0],
                            [14.0, 2.0, 1.0],
                        ],
                    ],
                ),
            ),
            FDataGrid(
                grid_points=np.array([0.75, 2.25]),
                data_matrix=np.array(
                    [
                        [
                            [1.5, 5.5, 9.5],
                            [3.5, 7.5, 0.5],
                        ],
                        [
                            [11.5, 6.0, 1.5],
                            [13.5, 1.5, 0.5],
                        ],
                    ],
                ),
            ),
        ],
        [
            FDataIrregular(
                start_indices=[0, 2],
                points=[
                    [1.0],
                    [4.0],
                    [1.0],
                    [3.0],
                    [5.0],
                ],
                values=[
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [4.0, 4.0, 1.0],
                    [5.0, 5.0, 1.0],
                ],
            ),
            FDataGrid(
                grid_points=np.array([2, 4]),
                data_matrix=np.array(
                    [
                        [
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 1.0],
                        ],
                        [
                            [3.0, 3.0, 1.0],
                            [4.5, 4.5, 1.0],
                        ],
                    ],
                ),
            ),
        ],
    ],
)
def precalc_example_data_domain_1(
    request: Any,
) -> list[FData, FDataGrid]:
    """
    Fixture for getting FData objects and their expected binned results.

    Calculations for these FDataGrid, FDataIrregular objects have been done
    previously and the expected result is returned as well.

    Dimensions interacting are 1->n.
    """
    return request.param


@pytest.fixture(
    params=[
        [
            FDataGrid(
                grid_points=[
                    [1, 2, 3],
                    # If there is one point but more than one bin, all points
                    # fall in the last bin and this behaviour is expected.
                    [1],
                    [1, 2, 3],
                ],
                data_matrix=np.array(
                    [
                        [  # First sample
                            [  # First x-index (1)
                                # (y=1) for all z values
                                [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
                            ],
                            [  # Second x-index (2)
                                [[1.3, 2.3], [1.4, 2.4], [1.5, 2.5]],
                            ],
                            [  # Third x-index (3)
                                [[1.6, 2.6], [1.7, 2.7], [1.8, 2.8]],
                            ],
                        ],
                        [  # Second sample
                            [  # First x-index (1)
                                [[2.0, 3.0], [2.1, 3.1], [2.2, 3.2]],
                            ],
                            [  # Second x-index (2)
                                [[2.3, 3.3], [2.4, 3.4], [2.5, 3.5]],
                            ],
                            [  # Third x-index (3)
                                [[2.6, 3.6], [2.7, 3.7], [2.8, 3.8]],
                            ],
                        ],
                    ],
                ),
            ),
            FDataGrid(
                # If only one value but more than one bin, duplicates the value
                # with the first ones being filled with nan values.
                grid_points=[
                    [1.5, 2.5],
                    [1.0, 1.0],
                    [1.5, 2.5],
                ],
                data_matrix=np.array(
                    [
                        [
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[1.0, 2.0], [1.15, 2.15]],
                            ],
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[1.45, 2.45], [1.6, 2.6]],
                            ],
                        ],
                        [
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[2.0, 3.0], [2.15, 3.15]],
                            ],
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[2.45, 3.45], [2.6, 3.6]],
                            ],
                        ],
                    ],
                ),
            ),
        ],
        [
            FDataIrregular(
                start_indices=[0, 4],
                points=[
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 1.0],
                    [2.5, 3.0, 3.0],
                    [4.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0],
                    [1.0, 4.0, 4.0],
                    [3.0, 3.0, 2.0],
                ],
                values=[
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ],
            ),
            # x = 2.5, y = 3, z = 2.5
            FDataGrid(
                grid_points=[
                    [1.75, 3.25],
                    [2.5, 3.5],
                    [1.75, 3.25],
                ],
                data_matrix=np.array(
                    [
                        [
                            [
                                [[3.0, 4.0], [1.0, 2.0]],
                                [[np.nan, np.nan], [np.nan, np.nan]],
                            ],
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[np.nan, np.nan], [6.0, 7.0]],
                            ],
                        ],
                        [
                            [
                                [[np.nan, np.nan], [1.0, 2.0]],
                                [[np.nan, np.nan], [3.0, 4.0]],
                            ],
                            [
                                [[np.nan, np.nan], [np.nan, np.nan]],
                                [[5.0, 6.0], [np.nan, np.nan]],
                            ],
                        ],
                    ],
                ),
            ),
        ],
    ],
)
def precalc_example_data_domain_3(
    request: Any,
) -> list[FData, FDataGrid]:
    """
    Fixture for getting FData objects and their expected binned results.

    Calculations for these FDataGrid, FDataIrregular objects have been done
    previously and the expected result is returned as well.

    Dimensions interacting are 3->n.
    """
    return request.param


##############################################################################
# Tests
##############################################################################


def test_raises(databinner_raises_fixture: BinnerFixture) -> None:
    """
    Check raises ValueError.

    Check that DataBinners raise a ValueError exception.
    """
    bins, domain_range_, output_grid, *rest = databinner_raises_fixture
    bin_aggregation, error_msg = rest

    with pytest.raises(ValueError, match=error_msg):
        DataBinner(
            bins=bins,
            domain_range=domain_range_,
            output_grid=output_grid,
            bin_aggregation=bin_aggregation,
        )


def test_domain_range_interactions(
    domain_range_interactions_fixture: list[FData, DataBinner, tuple],
) -> None:
    """
    Check the domain parameter behaviour.

    Check that the domain is correctly calculated for different combinations
    of domain_range, bins and domain.
    """
    fd, binner, expected_domain = domain_range_interactions_fixture
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


def test_precalc_example_domain_1(
    precalc_example_data_domain_1: list[FData, FDataGrid],
) -> None:
    """
    Check the precalculated example for binned FData.

    Compare the theoretical precalculated results against the binned data
    with DataBinner implementation, for different FData.

    Dimensions interacting are 1->n.
    """
    fd, precalc_result = precalc_example_data_domain_1
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


def test_precalc_example_domain_3(
    precalc_example_data_domain_3: list[FData, FDataGrid],
) -> None:
    """
    Check the precalculated example for binned FData.

    Compare the theoretical precalculated results against the binned data
    with DataBinner implementation, for different FData.

    Dimensions interacting are 3->n.
    """
    fd, precalc_result = precalc_example_data_domain_3
    binner = DataBinner(bins=(2, 2, 2))
    computed_result = binner.fit_transform(fd)

    np.testing.assert_allclose(
        precalc_result.data_matrix,
        computed_result.data_matrix,
        equal_nan=True,
    )

    np.testing.assert_array_equal(
        precalc_result.grid_points,
        computed_result.grid_points,
    )
