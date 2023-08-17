"""Typing for norms and metrics."""
from abc import abstractmethod
from typing import TypeVar

from typing_extensions import Protocol

from ._base import Vector
from ._numpy import NDArrayFloat

VectorType = TypeVar("VectorType", contravariant=True, bound=Vector)
MetricElementType = TypeVar("MetricElementType", contravariant=True)


class Norm(Protocol[VectorType]):
    """Protocol for a norm of a vector."""

    @abstractmethod
    def __call__(self, __vector: VectorType) -> NDArrayFloat:  # noqa: WPS112
        """Compute the norm of a vector."""


class Metric(Protocol[MetricElementType]):
    """Protocol for a metric between two elements of a metric space."""

    @abstractmethod
    def __call__(
        self,
        __e1: MetricElementType,  # noqa: WPS112
        __e2: MetricElementType,  # noqa: WPS112
    ) -> NDArrayFloat:
        """Compute the metric between two vectors."""
