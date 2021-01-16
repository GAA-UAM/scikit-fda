"""Common types."""
from typing import Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

VectorType = TypeVar("VectorType")

DomainRange = Tuple[Tuple[float, float], ...]
DomainRangeLike = Union[
    DomainRange,
    Sequence[float],
    Sequence[Sequence[float]],
]

LabelTuple = Tuple[Optional[str], ...]
LabelTupleLike = Sequence[Optional[str]]

GridPoints = Tuple[np.ndarray, ...]
GridPointsLike = Sequence[np.ndarray]


class Vector(Protocol):
    """
    Protocol representing a generic vector.

    It should accept numpy arrays and FData, among other things.
    """

    def __add__(self: VectorType, __other: VectorType) -> VectorType:
        pass

    def __sub__(self: VectorType, __other: VectorType) -> VectorType:
        pass

    def __mul__(self: VectorType, __other: float) -> VectorType:
        pass
