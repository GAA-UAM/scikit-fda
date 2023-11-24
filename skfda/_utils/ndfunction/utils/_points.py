"""Public utilities for dealing with arrays of functions."""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from .._array_api import Array, DType, NestedArray, Shape, array_namespace
from ..typing import GridPoints
from .validation import check_grid_points

A = TypeVar('A', bound=Array[Shape, DType])

if TYPE_CHECKING:
    from .._ndfunction import NDFunction


def cartesian_product(  # noqa: WPS234
    coords: NestedArray[A],
) -> A:
    r"""
    Compute the Cartesian product of the coordinates.

    Computes the Cartesian product of the coordinates and returns an array
    with all the possible combinations, for an arbitrary number of
    dimensions.

    Args:
        coords: Ragged array containing the coordinate values.

    Returns:
        Array with all the possible combinations, of shape
        (n_combinations, \*input_shape).

    Examples:
        We need to manually ensure that we create an array of arrays.

        >>> from skfda._utils.ndfunction.utils import cartesian_product
        >>> import numpy as np
        >>> coords = np.empty(shape=(2), dtype=np.object_)
        >>> coords[...] = [
        ...     np.array([0,1]),
        ...     np.array([2,3]),
        ... ]
        >>> cartesian_product(coords)
        array([[ 0, 2],
               [ 0, 3],
               [ 1, 2],
               [ 1, 3]])

        >>> coords = np.empty(shape=(3), dtype=np.object_)
        >>> coords[...] = [
        ...     np.array([0,1]),
        ...     np.array([2,3]),
        ...     np.array([4]),
        ... ]
        >>> cartesian_product(coords)
        array([[ 0, 2, 4],
               [ 0, 3, 4],
               [ 1, 2, 4],
               [ 1, 3, 4]])

        Higher dimensional arrays are also supported.

        >>> coords = np.empty(shape=(2, 2), dtype=np.object_)
        >>> coords[...] = [
        ...     [np.array([0,1]), np.array([2])],
        ...     [np.array([3]), np.array([4, 5, 6])],
        ... ]
        >>> cartesian_product(coords)
        array([[[ 0, 2],
                [ 3, 4]],
               [[ 0, 2],
                [ 3, 5]],
               [[ 0, 2],
                [ 3, 6]],
               [[ 1, 2],
                [ 3, 4]],
               [[ 1, 2],
                [ 3, 5]],
               [[ 1, 2],
                [ 3, 6]]])
    """
    coords = check_grid_points(coords)

    # The structure containing the arrays may be of different type,
    # e.g.: a NumPy array containing Pytorch tensors.
    # We currently assume this also follows the standard.
    xp_nested = array_namespace(coords)

    coords_flattened = xp_nested.reshape(
        coords,
        shape=-1,
    )

    # Warning: tolist not in the standard!!!
    xp = array_namespace(coords_flattened.tolist()[0])

    cartesian = xp.stack(
        xp.meshgrid(*coords_flattened, indexing='ij'),
        axis=-1,
    )

    return xp.reshape(  # type: ignore[no-any-return]
        cartesian,
        (-1,) + coords.shape,
    )


def grid_points_equal(gp1: GridPoints[A], gp2: GridPoints[A], /) -> bool:
    """Check if grid points are equal."""
    shape_equal = gp1.shape == gp2.shape
    values_equal = all(
        np.array_equal(arr1, arr2) for arr1, arr2 in zip(gp1.flat, gp2.flat)
    )

    return shape_equal and values_equal


def input_points_batch_shape(
    input_points: A,
    ndfunction: NDFunction[A],
    *,
    aligned: bool,
) -> tuple[int, ...]:
    """
    Retrieve the batch shape of input points.

    The shape of input points can be separated in three parts:
    - A leading shape identical to the NDFunction ``shape`` for unaligned
    evaluation.
    - A middle part containing the shape of the points themselves (for
    batch evaluation).
    - A final part with the same shape as the ``input_shape`` of the
    NDFunction.

    """
    shape = input_points.shape[:-len(ndfunction.input_shape)]
    if not aligned:
        shape = shape[len(ndfunction.shape):]

    return shape
