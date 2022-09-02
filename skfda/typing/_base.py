"""Common types."""
from typing import Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from ._numpy import ArrayLike, NDArrayFloat

VectorType = TypeVar("VectorType")

DomainRange = Tuple[Tuple[float, float], ...]
DomainRangeLike = Union[
    DomainRange,
    Sequence[float],
    Sequence[Sequence[float]],
]

LabelTuple = Tuple[Optional[str], ...]
LabelTupleLike = Sequence[Optional[str]]

GridPoints = Tuple[NDArrayFloat, ...]
GridPointsLike = Union[ArrayLike, Sequence[ArrayLike]]

EvaluationPoints = NDArrayFloat


RandomStateLike = Union[int, np.random.RandomState, np.random.Generator, None]
RandomState = Union[np.random.RandomState, np.random.Generator]


class Vector(Protocol):
    """
    Protocol representing a generic vector.

    It should accept numpy arrays and FData, among other things.
    """

    def __add__(
        self: VectorType,
        __other: VectorType,  # noqa: WPS112
    ) -> VectorType:
        pass

    def __sub__(
        self: VectorType,
        __other: VectorType,  # noqa: WPS112
    ) -> VectorType:
        pass

    def __mul__(
        self: VectorType,
        __other: float,  # noqa: WPS112
    ) -> VectorType:
        pass
