"""Types and functions associated with the Python array API standard."""
from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypeVar

import array_api_compat
import array_api_compat.numpy
import numpy as np
import numpy.typing
from typing_extensions import Self, TypeIs

DType = np.generic
NumericDType = np.number[Any]
RealDtype = np.integer[Any] | np.floating[Any]
D = TypeVar("D", bound=DType)
Shape: TypeAlias = Any
S = TypeVar("S", bound=Shape)
Array: TypeAlias = np.ndarray[S, np.dtype[D]]
BoolDType = np.bool_
A = TypeVar("A", bound=Array[Shape, DType])
ArrayLike: TypeAlias = np.typing.ArrayLike

numpy_namespace = array_api_compat.numpy
ArrayNamespace: TypeAlias[A] = Any


class NestedArray(Protocol[A]):  # type: ignore [misc]
    """Protocol representing an array of arrays."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array, not including ragged dimensions."""

    @property
    def ndim(self) -> int:
        """Return the number of non-ragged dimensions."""

    def __getitem__(
        self,
        key: int | slice | tuple[int | slice, ...],
        /,
    ) -> Self:
        pass

    def item(self) -> A:
        """Convert to Python representation."""


def array_namespace(
    *args: A | NestedArray[A],
) -> ArrayNamespace[A]:
    return (  # type: ignore [no-any-return]
        array_api_compat.array_namespace(*args)
    )


def is_array_api_obj(
    x: A | NestedArray[A] | object,
) -> TypeIs[A | NestedArray[A]]:
    """Check if x is an array API compatible array object."""
    return array_api_compat.is_array_api_obj(x)  # type: ignore [no-any-return]


def is_nested_array(
    array: A | NestedArray[A],
) -> TypeIs[NestedArray[A]]:
    """Check if an object is a nested array or a normal one."""
    # We use slices as using integers currently does not return
    # a 0D array in NumPy.

    first_pos = (
        array[(slice(0, 1),) * array.ndim]
        if array.ndim > 0 else array
    )

    try:
        first_element = first_pos.item()
        return is_array_api_obj(first_element)
    except AttributeError:
        return False
