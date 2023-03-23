"""Test the basic methods of the FDataIrregular structure"""
from typing import Tuple
from ..typing._numpy import ArrayLike
import numpy as np
import pandas
import pytest

from skfda.datasets._real_datasets import _fetch_loon_data
from skfda.representation import FDataIrregular, FDataGrid
from skfda.representation.interpolation import SplineInterpolation

############
# FIXTURES
############

NUM_CURVES = 10
DIMENSIONS = 2
TEST_DECIMALS = range(10)
COPY_KWARGS = [
    {"domain_range": ((0, 10))},
    {"dataset_name": "test"},
    {"sample_names": ["test"]*NUM_CURVES},
    # TODO Extrapolation
    {"interpolation": SplineInterpolation(3)},
    {"argument_names": ("test",)},
    {"coordinate_names": ("test",)},
]

random_state = np.random.RandomState(seed=14)

@pytest.fixture()
def input_arrays(
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generate three unidimensional arrays describing a
    FDataIrregular structure
    """
    # TODO Make editable with pytest
    num_values_per_curve = np.array(range(NUM_CURVES)) + 1

    values_per_curve = [random_state.rand(num_values, 1)
                        for num_values in num_values_per_curve]
    args_per_curve = [random_state.rand(num_values, 1)
                      for num_values in num_values_per_curve]
    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_multidimensional(
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generate three multidimensional arrays
    describing a FDataIrregular structure
    """
    # TODO Make editable with pytest
    num_values_per_curve = np.array(range(NUM_CURVES)) + 1

    values_per_curve = [random_state.rand(num_values, DIMENSIONS)
                        for num_values in num_values_per_curve]
    args_per_curve = [random_state.rand(num_values, DIMENSIONS)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def fdatagrid(
) -> FDataGrid:
    """Generate FDataGrid"""
    # TODO Make editable with pytest
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(NUM_CURVES, num_values_per_curve, 1)
    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(data_matrix=data_matrix,
                     grid_points=grid_points,
                     )


@pytest.fixture()
def fdatagrid_multidimensional(
) -> FDataGrid:
    """Generate multidimensional FDataGrid"""
    # TODO Make editable with pytest
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(NUM_CURVES, num_values_per_curve, DIMENSIONS)
    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(data_matrix=data_matrix,
                     grid_points=grid_points,
                     )


@pytest.fixture()
def dataframe(
) -> pandas.DataFrame:
    """Generate long dataframe for testing"""
    raw_dataset = _fetch_loon_data("bone_ext")
    data = raw_dataset["bone_ext"]

    return data

############
# TESTS
############


def test_fdatairregular_from_arrays(
    input_arrays: ArrayLike,
) -> None:
    """Tests creating a correct FDataIrregular object from escriptive arrays

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    assert f_data_irreg is not None
    assert len(f_data_irreg) == len(indices)
    assert len(f_data_irreg.function_arguments) == len(arguments)


def test_fdatairregular_from_multidimensional_arrays(
    input_arrays_multidimensional: ArrayLike,
) -> None:
    """Tests creating a correct FDataIrregular object from escriptive arrays

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays_multidimensional

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    assert f_data_irreg is not None
    assert len(f_data_irreg) == len(indices)
    assert len(f_data_irreg.function_arguments) == len(arguments)


def test_fdatairregular_copy(
    input_arrays: ArrayLike,
) -> None:
    """Test the copy function for FDataIrregular for an exact copy

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    assert f_data_irreg == f_data_irreg.copy()


@pytest.mark.parametrize("kwargs", COPY_KWARGS)
def test_fdatairregular_copy_kwargs(
    input_arrays: ArrayLike,
    kwargs: dict,
) -> None:
    """Test the copy function for FDataIrregular with additional arguments
    which replace certain parameters of the object

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
        kwargs: Dict with the parameters for each iteration of the test
    """
    indices, arguments, values = input_arrays

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    f_data_copy = f_data_irreg.copy(**kwargs)

    # Check everything equal except specified kwarg
    assert len(f_data_copy) == len(f_data_irreg)
    assert len(f_data_copy.function_arguments) == \
        len(f_data_irreg.function_arguments)
    assert f_data_copy.dim_domain == f_data_irreg.dim_domain
    assert f_data_copy.dim_domain == f_data_irreg.dim_codomain
    changed_attribute = next(iter(kwargs))
    assert getattr(f_data_copy, changed_attribute) != \
        getattr(f_data_irreg, changed_attribute)


def test_fdatairregular_from_fdatagrid(
    fdatagrid: FDataGrid,
) -> None:
    """Tests creating a correct FDataIrregular object from FDataGrid

    Args:
        fdatagrid (FDataGrid): FDataGrid object. Can be dense or sparse
        (contain NaNs)
    """
    f_data_irreg = FDataIrregular.from_datagrid(fdatagrid)

    assert f_data_irreg is not None
    assert len(f_data_irreg) == len(fdatagrid)


def test_fdatairregular_from_fdatagrid_multidimensional(
    fdatagrid_multidimensional: FDataGrid,
) -> None:
    """Tests creating a correct FDataIrregular object from
    a multidimensional FDataGrid

    Args:
        fdatagrid (FDataGrid): FDataGrid object. Can be dense or sparse
        (contain NaNs)
    """
    f_data_irreg = FDataIrregular.from_datagrid(fdatagrid_multidimensional)

    assert f_data_irreg is not None
    assert len(f_data_irreg) == len(fdatagrid_multidimensional)


def test_fdatairregular_from_dataframe(
    dataframe: FDataGrid,
) -> None:
    """Tests creating a correct FDataIrregular object from
    a multidimensional FDataGrid

    Args:
        fdatagrid (FDataGrid): FDataGrid object. Can be dense or sparse
        (contain NaNs)
    """

    curve_name = "idnum"
    argument_name = "age"
    coordinate_name = "spnbmd"

    f_irreg = FDataIrregular.from_dataframe(
        dataframe,
        id_column=curve_name,
        argument_columns=argument_name,
        coordinate_columns=coordinate_name,
        argument_names=[argument_name],
        coordinate_names=[coordinate_name],
        dataset_name="bone_ext"
    )

    assert len(f_irreg) == 423
    assert len(f_irreg.function_values) == 1003


def test_fdatairregular_getitem(
    input_arrays: ArrayLike,
) -> None:
    """Tests using slices to get subsamples of a given FDataIrregular,
    using the method __getitem__ of the class

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    assert len(f_data_irreg[0]) == 1
    assert len(f_data_irreg[-1]) == 1
    assert len(f_data_irreg[0:NUM_CURVES]) == NUM_CURVES
    assert len(f_data_irreg[0:]) == len(f_data_irreg)
    assert len(f_data_irreg[:NUM_CURVES]) == NUM_CURVES
    assert len(f_data_irreg[0:NUM_CURVES:2]) == NUM_CURVES/2
    assert len(f_data_irreg[0:NUM_CURVES:2]) == NUM_CURVES/2


def test_fdatairregular_coordinates(
    input_arrays_multidimensional: ArrayLike,
) -> None:
    """Test obtaining the different coordinates for a multidimensional
    FDataGrid object by using the custom _IrregularCoordinateIterator

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays_multidimensional

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    for dim, f_data_coordinate in enumerate(f_data_irreg.coordinates):
        assert len(f_data_coordinate) == len(f_data_irreg)
        assert f_data_coordinate.dim_codomain == 1
        assert f_data_coordinate.function_values[:, 0] == \
            f_data_irreg.function_values[:, dim]


@pytest.mark.parametrize("decimals", TEST_DECIMALS)
def test_fdatairregular_round(
    input_arrays: ArrayLike,
    decimals: int,
) -> None:
    """Test the round function for FDataIrregular

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays

    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )

    assert np.all(
        f_data_irreg.round(decimals).function_values ==
        np.round(f_data_irreg.function_values, decimals)
        )

def test_fdatairregular_equals(
    input_arrays: ArrayLike,
    input_arrays_multidimensional: ArrayLike,
) -> None:
    """Test for equals method, which in turn uses _eq_elementwise
    to verify equality in every index, argument and value

    Args:
        input_arrays (ArrayLike): tuple of three arrays required for
        FDataIrregular
        input_arrays_multidimensional (ArrayLike): tuple of three arrays required for
        FDataIrregular, with multiple dimensions
            indices: Array of pointers to the beginning of the arguments and
            values of each curve
            arguments: Array of each of the points of the domain
            values: Array of each of the coordinates of the codomain
    """
    indices, arguments, values = input_arrays_multidimensional

    f_data_irreg_multidimensional = FDataIrregular(
        indices,
        arguments,
        values,
        )
    
    indices, arguments, values = input_arrays
    
    f_data_irreg = FDataIrregular(
        indices,
        arguments,
        values,
        )
    
    assert f_data_irreg.equals(f_data_irreg)
    assert f_data_irreg_multidimensional.equals(f_data_irreg_multidimensional)
    assert not f_data_irreg.equals(f_data_irreg_multidimensional)
    assert f_data_irreg.equals(f_data_irreg.copy())