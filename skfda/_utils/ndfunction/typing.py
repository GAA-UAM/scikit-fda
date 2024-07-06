"""Type aliases and utilities for NDFunctions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar

import numpy as np

from ._array_api import Array, DType, NestedArray, Shape

A = TypeVar("A", bound=Array[Shape, DType])

GridPoints: TypeAlias = NestedArray
GridPointsLike: TypeAlias = A | Sequence[A] | NestedArray[A]

_FunctionNames: TypeAlias = np.ndarray[Shape, np.dtype[np.dtypes.StringDType]]
_FunctionNamesLike: TypeAlias = None | str | Sequence[str] | _FunctionNames

InputNames: TypeAlias = _FunctionNames
InputNamesLike: TypeAlias = _FunctionNamesLike

OutputNames: TypeAlias = _FunctionNames
OutputNamesLike: TypeAlias = _FunctionNamesLike
