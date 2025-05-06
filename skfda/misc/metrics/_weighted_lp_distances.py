"""Implementation of Weighted Lp distances."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ...representation import FData
from ...typing._numpy import NDArrayFloat
from ._utils import NormInducedMetric
from ._weighted_lp_norm import WeightedLpNorm

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...typing._base import (
        GridPointsLike,
    )
    from ...typing._metric import Norm

T = TypeVar("T", NDArrayFloat, FData)


class WeightedLpDistance(
    NormInducedMetric[NDArrayFloat | FData],
):
    def __init__(
        self,
        p: float,
        vector_norm: Norm[NDArrayFloat] | float | None = None,
        lp_weight: (
            Callable[[GridPointsLike], NDArrayFloat] | float | None
        ) = None,
    ) -> None:

        self.p = p
        self.vector_norm = vector_norm
        self.lp_weight = lp_weight
        norm = WeightedLpNorm(
            p=p, vector_norm=vector_norm, lp_weight=lp_weight,
        )

        super().__init__(norm)

    # This method is retyped here to work with either arrays or functions
    def __call__(self, elem1: T, elem2: T) -> NDArrayFloat:
        return super().__call__(elem1, elem2)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(p={self.p},"
            f" vector_norm={self.vector_norm})"
            f" lp_weight={self.lp_weight})"
        )


def weighted_lp_distance(
    fdata1: T,
    fdata2: T,
    *,
    p: float,
    vector_norm: Norm[NDArrayFloat] | float | None = None,
    lp_weight: Callable[[GridPointsLike], NDArrayFloat] | float | None = None,
) -> NDArrayFloat:
    return WeightedLpDistance(
        p=p,
        vector_norm=vector_norm,
        lp_weight=lp_weight,
    )(
        fdata1,
        fdata2,
    )
