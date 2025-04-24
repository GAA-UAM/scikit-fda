"""Typing for norms and metrics."""
import enum
from builtins import isinstance  # noqa: A004, UP029
from typing import Any, TypeVar, Union, overload

from typing_extensions import Final, Literal  # noqa: UP035

from ...typing._metric import Metric


class _MetricSingletons(enum.Enum):
    PRECOMPUTED = "precomputed"


PRECOMPUTED: Final = _MetricSingletons.PRECOMPUTED

_PrecomputedTypes = Literal[
    _MetricSingletons.PRECOMPUTED,
    "precomputed",
]

_NonStringMetric = TypeVar(
    "_NonStringMetric",
    bound=Union[  # noqa: UP007
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
    metric: Union[Metric[Any], _MetricSingletons, str],  # noqa: UP007
) -> Union[Metric[Any], _MetricSingletons]:  # noqa: UP007

    return _MetricSingletons(metric) if isinstance(metric, str) else metric
