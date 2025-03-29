"""Routines for input validation and conversion."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from .._array_api import (
    Array,
    ArrayLike,
    DType,
    Shape,
    array_namespace,
    is_array_api_obj,
    is_nested_array,
)
from .._region import AxisAlignedBox, Region

if TYPE_CHECKING:
    from ..typing import (
        GridPoints,
        GridPointsLike,
        InputNames,
        InputNamesLike,
        OutputNames,
        OutputNamesLike,
        RegionLike,
        _FunctionNames,
        _FunctionNamesLike,
    )

A = TypeVar("A", bound=Array[Shape, DType])


def check_grid_points(grid_points_like: GridPointsLike[A]) -> GridPoints[A]:
    """
    Convert to grid points.

    Grid points are represented as an "array of arrays", containing at each
    position the array of grid points for that position.

    A sequence of arrays would be converted to that representation, replacing
    the sequence by a unidimensional array.

    If an array is received, it is processed as a sequence of just one element.

    Args:
        grid_points_like: Grid points as an "array of arrays", sequence of
            arrays, or just an array.

    Returns:
        Grid points as an "array of arrays".

    """
    if is_array_api_obj(grid_points_like):
        if is_nested_array(grid_points_like):
            return grid_points_like

        # It is an array
        grid_points = np.empty(shape=1, dtype=np.object_)
        grid_points[0] = grid_points_like
        return np.squeeze(grid_points)

    # It is a sequence!
    # Ensure that elements are compatible arrays
    array_namespace(*grid_points_like)
    grid_points = np.empty(shape=len(grid_points_like), dtype=np.object_)
    grid_points[...] = grid_points_like
    return grid_points


def check_evaluation_points(
    eval_points: A,
    *,
    aligned: bool,
    shape: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> A:
    """
    Check the evaluation points.

    The leading dimensions need to be the same as the array shape in the
    unaligned case.
    The trailing dimensions of the shape of the evaluation points need to be
    the same as the input shape.

    Args:
        eval_points: Evaluation points to be reshaped.
        aligned: Boolean flag. True if all the samples
            will be evaluated at the same evaluation_points.
        shape: Shape of the array of functions.
        input_shape: Shape of the input accepted by the functions.

    Returns:
        Evaluation points if all checks pass. Otherwise an exception is raised.

    """
    if not aligned and eval_points.shape[:len(shape)] != shape:
        msg = (
            f"Invalid shape for evaluation points."
            f"The leading shape dimensions in the unaligned case "
            f"were expected to be {shape}, corresponding with the "
            f"shape of the array."
            f"Instead, the received evaluation points have shape "
            f"{eval_points.shape}."
        )
        raise ValueError(msg)

    if eval_points.shape[-len(input_shape):] != input_shape:

        # This should probably be removed in the future.
        if input_shape == (1,):
            # Add a new dimension
            eval_points = eval_points[..., None]
        else:
            msg = (
                f"Invalid shape for evaluation points."
                f"The trailing shape dimensions were expected to be "
                f"{input_shape}, corresponding with the input shape."
                f"Instead, the received evaluation points have shape "
                f"{eval_points.shape}."
            )
            raise ValueError(msg)

    return eval_points


def _arraylike_conversion(
    array: ArrayLike,
    *,
    namespace: Any,
    allow_array_like: bool = False,
) -> Array[Shape, DType]:
    if allow_array_like:
        return namespace.asarray(array)  # type: ignore[no-any-return]

    msg = f"{type(array)} is not compatible with the array API standard."
    raise ValueError(msg)


@overload
def check_array_namespace(
    *args: A,
    namespace: Any,
    allow_array_like: Literal[False] = False,
) -> tuple[A, ...]:
    pass


@overload
def check_array_namespace(
    *args: A | ArrayLike,
    namespace: Any,
    allow_array_like: Literal[True],
) -> tuple[A, ...]:
    pass


def check_array_namespace(
    *args: A | ArrayLike,
    namespace: Any,
    allow_array_like: bool = False,
) -> tuple[A, ...]:
    """
    Check if the array namespace is appropriate.

    Args:
        args: Arrays to check.
        namespace: The namespace to check.
        allow_array_like: Whether array-likes are allowed.

    Returns:
        The input arrays as objects of the namespace.

    """
    converted: list[A] = [
        array  # type: ignore[misc]
        if is_array_api_obj(array)
        else _arraylike_conversion(
            array,
            namespace=namespace,
            allow_array_like=allow_array_like,
        )
        for array in args
    ]

    return tuple(converted)


def _check_function_names(
    names: _FunctionNamesLike,
    shape: tuple[int, ...],
    names_type: Literal["input", "output"],
) -> _FunctionNames:
    """
    Convert to proper input/output names.

    Input/output names are a string array, broadcastable to ``shape``.
    A string or a sequence of strings will be accepted and converted.

    Args:
        names: The names of inputs or outputs of the function.
        shape: The corresponding shape of the input/output.
        names_type: Whether the input or output names are being checked.

    Returns:
        The string array containing the names.

    """
    # TODO: Change when https://github.com/numpy/numpy/issues/28609 is fixed.
    # dtype = np.dtypes.StringDType(na_object=np.nan)
    # default_value = np.nan
    dtype = np.str_
    default_value = ""
    if isinstance(names, np.ndarray):
        names = names.astype(dtype)
    else:
        old_names = default_value if names is None else names
        names = np.array(old_names, dtype=dtype)

    # Check that it can be broadcasted
    if np.broadcast_shapes(names.shape, shape) != shape:
        msg = (
            f"The {names_type} names have shape {names.shape}, which is not "
            f"broadcastable to the {names_type} shape ({shape})."
        )
        raise ValueError(msg)

    return names


def check_input_names(
    names: InputNamesLike,
    shape: tuple[int, ...],
) -> InputNames:
    """
    Convert to proper input names.

    Input names are a string array, broadcastable to ``shape``.
    A string or a sequence of strings will be accepted and converted.

    Args:
        names: The names of inputs of the function.
        shape: The corresponding shape of the input.

    Returns:
        The string array containing the names.

    """
    return _check_function_names(
        names=names,
        shape=shape,
        names_type="input",
    )


def check_output_names(
    names: OutputNamesLike,
    shape: tuple[int, ...],
) -> OutputNames:
    """
    Convert to proper output names.

    Output names are a string array, broadcastable to ``shape``.
    A string or a sequence of strings will be accepted and converted.

    Args:
        names: The names of outputs of the function.
        shape: The corresponding shape of the output.

    Returns:
        The string array containing the names.

    """
    return _check_function_names(
        names=names,
        shape=shape,
        names_type="output",
    )


def check_region(
    region: RegionLike[A],
) -> Region[A]:
    """
    Convert to a proper region.

    Args:
        region: The value to convert to a region. If it is already a region,
            it is checked and returned. Otherwise, the input needs to be a
            tuple of two arrays.

    Returns:
        A valid region object with the desired shape and array type.

    """
    match region:
        case (lower, upper):
            return AxisAlignedBox(lower=lower, upper=upper)
        case _:
            if not isinstance(region, Region):
                msg = (
                    f"Expected an instance of an object that follows the "
                    f"Region protocol. Got {type(region)} instead."
                )
                raise TypeError(msg)

            return region
