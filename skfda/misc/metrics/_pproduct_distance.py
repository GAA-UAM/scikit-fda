
"""Implementation of Weighted Lp distances."""
from __future__ import annotations

from typing import TypeVar, Union

from ...representation import FData
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat
from ._pproduct_metric import PProductMetric
from ._utils import NormInducedMetric

T = TypeVar("T", NDArrayFloat, FData)


class PProductDistance(NormInducedMetric[Union[NDArrayFloat, FData]]):  # noqa: UP007
    def __init__(
        self,
        p: float,
        norms: list[Norm] | Norm | None = None,
        weights: NDArrayFloat | float | None = None,
    ) -> None:

        self.p = p
        self.norms = norms
        self.weights = weights
        norm = PProductMetric(p=p, norms=norms, weights=weights)

        super().__init__(norm)

    # This method is retyped here to work with either arrays or functions
    def __call__(self, elem1: T, elem2: T) -> NDArrayFloat:
        return super().__call__(elem1, elem2)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p={self.p}, norms={self.norms} ,weigths={self.weights})"
        )

def pproduct_distance(
    fdata1: T,
    fdata2: T,
    *,
    p:float,
    norms: list[Norm] | Norm | None = None,
    weights: NDArrayFloat | float | None = None,
) -> NDArrayFloat:
    return PProductDistance(p=p,norms=norms, weights=weights)(fdata1, fdata2)
