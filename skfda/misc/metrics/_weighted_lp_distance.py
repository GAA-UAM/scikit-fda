
"""Implementation of Weighted Lp distances."""
from __future__ import annotations

import math
from typing import Optional, TypeVar, Callable, Union

import numpy as np
from typing_extensions import Final

from ...representation import FData
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat
from ._weighted_lp_norm import WeightedLpNorm
from ._utils import NormInducedMetric, pairwise_metric_optimization

T = TypeVar("T", NDArrayFloat, FData)


class WeightedLpDistance(NormInducedMetric[Union[NDArrayFloat, FData]]):
    def __init__(
        self,
        p: float,
        vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
        lp_weight: Union[
            Callable[[NDArrayFloat], NDArrayFloat],
            float,
            None,
        ] = None,
    ) -> None:

        self.p = p
        self.vector_norm = vector_norm
        norm = WeightedLpNorm(p=p, vector_norm=vector_norm, lp_weight=lp_weight)

        super().__init__(norm)

    # This method is retyped here to work with either arrays or functions
    def __call__(self, elem1: T, elem2: T) -> NDArrayFloat:  # noqa: WPS612
        return super().__call__(elem1, elem2)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p={self.p}, vector_norm={self.vector_norm})"
        )



def lp_distance(
    fdata1: T,
    fdata2: T,
    *,
    p: float,
    vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
    lp_weight: Union[
        Callable[[NDArrayFloat], NDArrayFloat],
        float,
        None,
    ] = None,
) -> NDArrayFloat:
    return WeightedLpDistance(p=p,vector_norm=vector_norm, lp_weight=lp_weight)(fdata1, fdata2)
