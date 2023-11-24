"""Type aliases and utilities for NDFunctions."""
from __future__ import annotations

from typing import Sequence, TypeVar, Union

from typing_extensions import TypeAlias

from ._array_api import Array, DType, NestedArray, Shape

A = TypeVar('A', bound=Array[Shape, DType])

GridPoints: TypeAlias = NestedArray
GridPointsLike: TypeAlias = Union[A, Sequence[A], NestedArray[A]]
