"""Common types."""
from typing import Optional, Sequence, Tuple, TypeVar, Union  # noqa: UP035

import numpy as np
from typing_extensions import Protocol

from ._numpy import ArrayLike, NDArrayFloat

VectorType = TypeVar("VectorType")

DomainRange = Tuple[Tuple[float, float], ...]  # noqa: UP006
DomainRangeLike = Union[  # noqa: UP007
    DomainRange,
    Sequence[float],
    Sequence[Sequence[float]],
]

LabelTuple = Tuple[Optional[str], ...]  # noqa: UP006, UP007
LabelTupleLike = Sequence[Optional[str]]  # noqa: UP007

GridPoints = Tuple[NDArrayFloat, ...]  # noqa: UP006
GridPointsLike = Union[ArrayLike, Sequence[ArrayLike]]  # noqa: UP007

EvaluationPoints = NDArrayFloat


RandomStateLike = Union[int, np.random.RandomState, np.random.Generator, None]  # noqa: UP007
RandomState = Union[np.random.RandomState, np.random.Generator]  # noqa: UP007


class Vector(Protocol):
    """
    Protocol representing a generic vector.

    It should accept numpy arrays and FData, among other things.
    """

    def __add__(  # noqa: PYI019
        self: VectorType,
        __other: VectorType,  # noqa: PYI063, WPS112
    ) -> VectorType:
        pass

    def __sub__(  # noqa: PYI019
        self: VectorType,
        __other: VectorType,  # noqa: PYI063, WPS112
    ) -> VectorType:
        pass

    def __mul__(  # noqa: PYI019
        self: VectorType,
        __other: float,  # noqa: PYI063, WPS112
    ) -> VectorType:
        pass
