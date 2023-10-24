"""Test the basic methods of the FDataIrregular structure."""
from typing import Any, Tuple

import numpy as np
import pandas
import pytest

from skfda.datasets._real_datasets import _fetch_loon_data
from skfda.representation import FDataGrid, FDataIrregular
from skfda.representation.interpolation import SplineInterpolation

from ..typing._numpy import ArrayLike

############
# FIXTURES
############

SEED = 2906198114

NUM_CURVES = 10
DIMENSIONS = 2
TEST_DECIMALS = range(10)
COPY_KWARGS = [  # noqa: WPS407
    {"domain_range": ((0, 10))},
    {"dataset_name": "test"},
    {"sample_names": ["test"] * NUM_CURVES},
    {"interpolation": SplineInterpolation(3)},
    {"argument_names": ("test",)},
    {"coordinate_names": ("test",)},
]

random_state = np.random.RandomState(seed=SEED)


@pytest.fixture()
def input_arrays(
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Create unidimensional arrays describing a FDataIrregular structure."""
    num_values_per_curve = np.array(range(NUM_CURVES)) + 1

    values_per_curve = [
        random_state.rand(num_values, 1)
        for num_values in num_values_per_curve
    ]

    args_per_curve = [
        random_state.rand(num_values, 1)
        for num_values in num_values_per_curve
    ]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_multidimensional(
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Create multidimensional arrays describing a FDataIrregular structure."""
    num_values_per_curve = np.array(range(NUM_CURVES)) + 1

    values_per_curve = [
        random_state.rand(num_values, DIMENSIONS)
        for num_values in num_values_per_curve
    ]

    args_per_curve = [
        random_state.rand(num_values, DIMENSIONS)
        for num_values in num_values_per_curve
    ]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture(
    params=[
        "input_arrays",
        "input_arrays_multidimensional",
    ],
)
def fdatairregular(
    request: Any,
    input_arrays: FDataIrregular,
    input_arrays_multidimensional: FDataIrregular,
) -> FDataIrregular:
    """Return 'input_arrays' or 'input_arrays_multidimensional'."""
    if request.param == "input_arrays":
        return FDataIrregular(*input_arrays)
    elif request.param == "input_arrays_multidimensional":
        return FDataIrregular(*input_arrays_multidimensional)


@pytest.fixture()
def fdatagrid_unidimensional(
) -> FDataGrid:
    """Generate FDataGrid."""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(NUM_CURVES, num_values_per_curve, 1)
    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )


@pytest.fixture()
def fdatagrid_multidimensional(
) -> FDataGrid:
    """Generate multidimensional FDataGrid."""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(
        NUM_CURVES,
        num_values_per_curve,
        DIMENSIONS,
    )

    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )


@pytest.fixture(
    params=[
        "fdatagrid_unidimensional",
        "fdatagrid_multidimensional",
    ],
)
def fdatagrid(
    request: Any,
    fdatagrid_unidimensional: FDataGrid,
    fdatagrid_multidimensional: FDataGrid,
) -> FDataIrregular:
    """Return 'fdatagrid_unidimensional' or 'fdatagrid_multidimensional'."""
    if request.param == "fdatagrid_unidimensional":
        return fdatagrid_unidimensional
    elif request.param == "fdatagrid_multidimensional":
        return fdatagrid_multidimensional


@pytest.fixture()
def dataframe(
) -> pandas.DataFrame:
    """Generate long dataframe for testing."""
    raw_dataset = _fetch_loon_data("bone_ext")

    return raw_dataset["bone_ext"]


@pytest.fixture(
    params=[
        "unidimensional",
        "multidimensional",
    ],
)
def fdatairregular_and_sum(request: Any) -> FDataIrregular:
    if request.param == "unidimensional":
        return (
            FDataIrregular(
                start_indices=[0, 3, 7],
                points=[
                    -9, -3, 3, -3, 3, 9, 15, -15, -9, -3, 3, 9, 17, 22, 29,
                ],
                values=[
                    548, 893, 657, 752, 459, 181, 434, 846, 1102, 801, 824,
                    866, 704, 757, 726,
                ],
            ),
            FDataIrregular(
                start_indices=[0],
                points=[-3, 3],
                values=[2446, 1940],
            ),
        )
    if request.param == "multidimensional":
        return (
            FDataIrregular(
                start_indices=[0, 3, 5],
                points=[
                    [0, 0], [1, 2], [1, 1],
                    [0, 0], [1, 1],
                    [0, 0], [6, 2], [1, 1],
                ],
                values=[
                    [0, 0, -1], [657, 752, 5], [10, 20, 30],
                    [-1, 0, 0], [1102, 801, 2],
                    [0, 1, 0], [704, 0, 757], [-11, -21, 31],
                ],
            ),
            FDataIrregular(
                start_indices=[0],
                points=[[0, 0], [1, 1]],
                values=[[-1, 1, -1], [1101, 800, 63]],
            ),
        )


@pytest.fixture()
def fdatairregular_no_common_points() -> FDataIrregular:
    return FDataIrregular(
        start_indices=[0, 3, 5],
        points=[
            [0, 1], [1, 2], [1, 1],
            [0, -1], [1, 10],
            [0, -2], [6, 2], [10, 1],
        ],
        values=[
            [0, 0, -1], [657, 752, 5], [10, 20, 30],
            [-1, 0, 0], [1102, 801, 2],
            [0, 1, 0], [704, 0, 757], [-11, -21, 31],
        ],
    )

############
# TESTS
############


def test_fdatairregular_init(
    fdatairregular: FDataIrregular,
) -> None:
    """Tests creating a correct FDataIrregular object from arrays.

    Test both unidimensional and multidimensional.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    arguments = fdatairregular.points
    assert fdatairregular is not None
    assert len(fdatairregular) == len(fdatairregular.start_indices)
    assert len(arguments) == len(fdatairregular.values)


def test_fdatairregular_copy(
    fdatairregular: FDataIrregular,
) -> None:
    """Test the copy function for FDataIrregular for an exact copy.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    assert np.all(fdatairregular == fdatairregular.copy())


@pytest.mark.parametrize("kwargs", COPY_KWARGS)
def test_fdatairregular_copy_kwargs(
    fdatairregular: FDataIrregular,
    kwargs: dict,
) -> None:
    """Test the copy function for FDataIrregular.

    Test with additional keyword arguments which replace
    certain parameters of the object.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
        kwargs: Dict with the parameters for each iteration of the test
    """
    changed_attribute = next(iter(kwargs))
    local_kwargs = kwargs.copy()

    if changed_attribute == "argument_names":
        # Set correct dimensionality
        dim = fdatairregular.dim_domain
        local_kwargs[changed_attribute] = kwargs[changed_attribute] * dim
    if changed_attribute == "coordinate_names":
        # Set correct dimensionality
        dim = fdatairregular.dim_codomain
        local_kwargs[changed_attribute] = kwargs[changed_attribute] * dim

    f_data_copy = fdatairregular.copy(**local_kwargs)

    og_attribute = getattr(fdatairregular, changed_attribute)
    copy_attribute = getattr(f_data_copy, changed_attribute)

    # Check everything equal except specified kwarg
    assert len(f_data_copy) == len(fdatairregular)
    assert len(f_data_copy.points) == len(fdatairregular.points)
    assert f_data_copy.dim_domain == fdatairregular.dim_domain
    assert f_data_copy.dim_domain == fdatairregular.dim_codomain
    assert og_attribute != copy_attribute


def test_fdatairregular_from_fdatagrid(
    fdatagrid: FDataGrid,
) -> None:
    """Tests creating a correct FDataIrregular object from FDataGrid.

    Args:
        fdatagrid (FDataGrid): FDataGrid object. Can be dense or sparse
            (contain NaNs)
    """
    f_data_irreg = FDataIrregular.from_fdatagrid(fdatagrid)

    assert f_data_irreg is not None
    assert len(f_data_irreg) == len(fdatagrid)


def test_fdatairregular_from_dataframe(
    dataframe: pandas.DataFrame,
) -> None:
    """Test creating FDataIrregular from pandas DataFrame.

    Args:
        dataframe (pandas:DataFrame): DataFrame object.
            It should be in 'long' format.
    """
    curve_name = "idnum"
    argument_name = "age"
    coordinate_name = "spnbmd"

    f_irreg = FDataIrregular._from_dataframe(
        dataframe,
        id_column=curve_name,
        argument_columns=argument_name,
        coordinate_columns=coordinate_name,
        argument_names=[argument_name],
        coordinate_names=[coordinate_name],
        dataset_name="bone_ext",
    )

    assert len(f_irreg) == 423
    assert len(f_irreg.values) == 1003


def test_fdatairregular_getitem(
    fdatairregular: FDataIrregular,
) -> None:
    """Tests the getitem method of FDataIrregular.

    Use slices to get subsamples of a given FDataIrregular,
    using the method __getitem__ of the class, and then
    verify the length of the result is correct.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    assert len(fdatairregular[0]) == len(fdatairregular[-1]) == 1
    assert len(fdatairregular[:]) == len(fdatairregular)
    assert len(fdatairregular[:NUM_CURVES]) == NUM_CURVES
    assert len(fdatairregular[:NUM_CURVES:2]) == NUM_CURVES / 2
    assert len(fdatairregular[:NUM_CURVES:2]) == NUM_CURVES / 2


def test_fdatairregular_coordinates(
    fdatairregular: FDataIrregular,
) -> None:
    """Test the coordinates function.

    First obtain the different coordinates for a multidimensional
    FDataGrid object by using the custom _IrregularCoordinateIterator.

    Then check that the coordinates are equal elementwise to the
    original.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    for dim, f_data_coordinate in enumerate(fdatairregular.coordinates):
        assert len(f_data_coordinate) == len(fdatairregular)
        assert f_data_coordinate.dim_codomain == 1
        assert np.all(
            f_data_coordinate.values[:, 0] == fdatairregular.values[:, dim],
        )


@pytest.mark.parametrize("decimals", TEST_DECIMALS)
def test_fdatairregular_round(
    fdatairregular: FDataIrregular,
    decimals: int,
) -> None:
    """Test the round function for FDataIrregular.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
        decimals (int): Number of decimal places to round.
    """
    assert np.all(
        fdatairregular.round(decimals).values
        == np.round(fdatairregular.values, decimals),
    )


def test_fdatairregular_concatenate(
    fdatairregular: FDataIrregular,
) -> None:
    """Test concatenate FDataIrregular objects.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    fd_concat = fdatairregular.concatenate(fdatairregular)

    start_indices_halves = np.split(fd_concat.start_indices, 2)
    indices = fdatairregular.start_indices
    second_half_indices = indices + len(fdatairregular.points)

    function_args_halves = np.split(fd_concat.points, 2)
    values_halves = np.split(fd_concat.values, 2)

    assert len(fd_concat) == 2 * len(fdatairregular)
    assert np.all(start_indices_halves[1] == second_half_indices)
    assert len(fd_concat.points) == 2 * len(fdatairregular.points)
    assert np.all(function_args_halves[1] == fdatairregular.points)
    assert np.all(values_halves[1] == fdatairregular.values)


def test_fdatairregular_equals(
    fdatairregular: FDataIrregular,
) -> None:
    """Test for equals method.

    It uses _eq_elementwise to verify equality in every
    index, argument and value.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    assert fdatairregular.equals(fdatairregular)
    assert fdatairregular.equals(fdatairregular.copy())


def test_fdatairregular_restrict(
    fdatairregular: FDataIrregular,
) -> None:
    """Test the restrict function for FDataIrregular.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    restricted_domain = [
        (dr[0] + (dr[0] + dr[1]) / 4, dr[1] - (dr[0] + dr[1]) / 4)
        for dr in fdatairregular.domain_range
    ]

    restricted_fdata = fdatairregular.restrict(restricted_domain)

    samples_by_dim = [
        restricted_fdata.points[:, dim]
        for dim in range(fdatairregular.dim_domain)
    ]

    sample_ranges = [(np.min(args), np.max(args)) for args in samples_by_dim]

    # The min arg is larger than the domain min constraint
    assert len(restricted_fdata) > 0
    assert all(
        sr[0] > restricted_domain[i][0]
        for i, sr in enumerate(sample_ranges)
    )

    # The max arg is lesser than the domain max constraint
    assert all(
        sr[1] < restricted_domain[i][1]
        for i, sr in enumerate(sample_ranges)
    )


def test_fdatairregular_to_grid(
    fdatairregular: FDataIrregular,
    fdatagrid: FDataGrid,
) -> None:
    """Test conversion of FDataIrregular to and from FDataGrid.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
        fdatagrid (FDataGrid): FDataGrid object.
    """
    f_data_grid = fdatairregular.to_grid()

    # FDataGrid -> FDataIrregular -> FDataGrid
    assert fdatagrid.equals(FDataIrregular.from_fdatagrid(fdatagrid).to_grid())
    # FDataIrregular -> FDataGrid -> FDataIrregular
    assert fdatairregular.equals(FDataIrregular.from_fdatagrid(f_data_grid))


def test_fdatairregular_isna(
    fdatairregular: FDataIrregular,
) -> None:
    """Test the shape of isna function output for FDataIrregular.

    Args:
        fdatairregular (FDataIrregular): FDataIrregular object
            which can be unidimensional or multidimensional.
    """
    assert fdatairregular.isna().shape == (len(fdatairregular),)


def test_fdatairregular_sum(
    fdatairregular_and_sum: Tuple[FDataIrregular, FDataIrregular],
) -> None:
    """Test the sum function for FDataIrregular.

    Test both unidimensional and multidimensional.

    Args:
        fdatairregular_and_sum: FDataIrregular object and expected sum.
    """
    fdatairregular, expected_sum = fdatairregular_and_sum
    actual_sum = fdatairregular.sum()
    assert actual_sum.equals(expected_sum), actual_sum


def test_fdatairregular_sum_invalid(
    fdatairregular_no_common_points: FDataIrregular,
) -> None:
    """Test the sum function for FDataIrregular, case with no common points.

    Args:
        fdatairregular_no_common_points: FDataIrregular object with no common
            points.
    """
    with pytest.raises(ValueError):
        fdatairregular_no_common_points.sum()
