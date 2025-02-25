"""Tests for DataBinner module."""
from typing import Any

import pytest
import numpy as np
from skfda.representation import FData, FDataGrid, FDataIrregular
from skfda.preprocessing.binning import DataBinner


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
            "For 2-dimensional domain, range must be a tuple with 2 "
            "tuples",
        ],
    ],
)
def databinner_raises_fixture(request: Any) -> Any:
    """Fixture for getting a DataBinner object that raises a ValueError"""
    return request.param 

@pytest.fixture(
    params = [
        [
            FDataGrid(
                grid_points=np.linspace(0,3,4),
                data_matrix=np.array([
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                ])
            ),
            FDataGrid(
                grid_points=np.array([0.75, 2.25]),
                data_matrix=np.array([
                    [1.5, 3.5],
                    [3.5, 1.5],
                ]),
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
    databinner_raises_fixture: Any,
) -> None:
    """
    Check raises ValueError.

    Check that DataBinners raise a ValueError exception.
    """
    bins, range_, output_grid, bin_aggregation, error_msg = databinner_raises_fixture

    with pytest.raises(ValueError, match=error_msg):
        DataBinner(
            bins=bins,
            range=range_,
            output_grid=output_grid,
            bin_aggregation=bin_aggregation,
        )


def test_precalc_example(
    precalc_example_data_1_1: list[FData, FDataGrid]
):
    """
    Check the precalculated example for binned FData.

    Compare the theoretical precalculated results against the binned data
    with DataBinner implementation, for different FData.
    """
    fd, precalc_result = precalc_example_data_1_1
    binner = DataBinner(bins=2)
    computed_result = binner.fit_transform(fd)
    # np.testing.assert_allclose(
    #     computed_result,
    #     precalc_result,
    #     rtol=1e-6,
    # )

    print(precalc_result)
    print(computed_result)
    np.testing.assert_array_equal(
        precalc_result.data_matrix,
        computed_result.data_matrix,
    )

    np.testing.assert_array_equal(
        precalc_result.grid_points,
        computed_result.grid_points,
    )

# def test_domain_range_interactions
# def test_mismatch_binner_and_fd
# def test_output_grid
# def test_calculation_mode