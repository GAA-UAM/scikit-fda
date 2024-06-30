"""Type aliases and utilities for NDFunctions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar

from ._array_api import Array, DType, NestedArray, Shape

A = TypeVar("A", bound=Array[Shape, DType])

GridPoints: TypeAlias = NestedArray
GridPointsLike: TypeAlias = A | Sequence[A] | NestedArray[A]
