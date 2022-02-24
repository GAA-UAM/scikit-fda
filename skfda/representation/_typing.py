"""Common types."""
from typing import Any, Optional, Sequence, Tuple, TypeVar, Union

from typing_extensions import Protocol

import numpy as np

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = np.ndarray  # type:ignore

try:
    from numpy.typing import NDArray
    NDArrayAny = NDArray[Any]
    NDArrayInt = NDArray[np.int_]
    NDArrayFloat = NDArray[np.float_]
    NDArrayBool = NDArray[np.bool_]
except ImportError:
    NDArray = np.ndarray  # type:ignore
    NDArrayAny = np.ndarray  # type:ignore
    NDArrayInt = np.ndarray  # type:ignore
    NDArrayFloat = np.ndarray  # type:ignore
    NDArrayBool = np.ndarray  # type:ignore

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
GridPointsLike = Union[ArrayLike, Sequence[ArrayLike]]


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
