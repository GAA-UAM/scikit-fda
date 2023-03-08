"""Test the basic methods of the FDataIrregular structure"""
from typing import Any, Callable, Tuple
from ..typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
import numpy as np
import pytest

from skfda.datasets import fetch_bone_density
from skfda.misc.covariances import CovarianceLike, Gaussian
from skfda.representation import FDataIrregular, FDataGrid

############
# FIXTURES
############

NUM_CURVES = 10
MAX_VALUES_PER_CURVE = 99
DIMENSIONS = 2


@pytest.fixture()
def input_arrays(
) -> ArrayLike:
    """Generate three unidimensional arrays describing a FDataIrregular structure"""
    # TODO Make editable with pytest
    num_curves = NUM_CURVES
    num_values_per_curve = np.random.randint(1,
                                             MAX_VALUES_PER_CURVE,
                                             size=(num_curves, )
                                             )

    values_per_curve = [np.random.rand(num_values, 1)
                        for num_values in num_values_per_curve]
    args_per_curve = [np.random.rand(num_values, 1)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_multidimensional(
) -> ArrayLike:
    """Generate three multidimensional arrays describing a FDataIrregular structure"""
    # TODO Make editable with pytest
    num_curves = NUM_CURVES
    num_values_per_curve = np.random.randint(1,
                                             MAX_VALUES_PER_CURVE,
                                             size=(num_curves, )
                                             )

    values_per_curve = [np.random.rand(num_values, DIMENSIONS)
                        for num_values in num_values_per_curve]
    args_per_curve = [np.random.rand(num_values, DIMENSIONS)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments

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
    
    
