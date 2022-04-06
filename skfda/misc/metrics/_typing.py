"""Typing for norms and metrics."""
import enum
from abc import abstractmethod
from builtins import isinstance
from typing import Any, TypeVar, Union, overload

from typing_extensions import Final, Literal, Protocol

from ...representation._typing import NDArrayFloat, Vector

VectorType = TypeVar("VectorType", contravariant=True, bound=Vector)
MetricElementType = TypeVar("MetricElementType", contravariant=True)


class _MetricSingletons(enum.Enum):
    PRECOMPUTED = "precomputed"


PRECOMPUTED: Final = _MetricSingletons.PRECOMPUTED

_PrecomputedTypes = Literal[
    _MetricSingletons.PRECOMPUTED,
    "precomputed",
]


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


_NonStringMetric = TypeVar(
    "_NonStringMetric",
    bound=Union[
        Metric[Any],
        _MetricSingletons,
    ],
)


@overload
def _parse_metric(
    metric: str,
) -> _MetricSingletons:
    pass


@overload
def _parse_metric(
    metric: _NonStringMetric,
) -> _NonStringMetric:
    pass


def _parse_metric(
    metric: Union[Metric[Any], _MetricSingletons, str],
) -> Union[Metric[Any], _MetricSingletons]:

    return _MetricSingletons(metric) if isinstance(metric, str) else metric
