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
    r"""
    Weighted Lp distance for functional data objects.

    Calculates the distance between pairs of functional observations using a
    weighted Lp norm. This class extends the standard Lp distance by
    introducing a weighting function over the domain, allowing certain regions
    to contribute more or less to the final distance.

    Given two functional observations \( x(t) \) and \( y(t) \), the weighted
    Lp distance is defined as:

    .. math::
        d_{p,w}(x, y) = \| x - y \|_{p,w} =
        \left( \int_{\mathcal{T}} w(t) \| x(t) - y(t) \|_{\mathbb{R}^D}^p \,
        dt \right)^{1/p},

    where:
        - \( w(t) \) is a non-negative weighting function over the domain \(
            \mathcal{T} \),
        - \( \| \cdot \|_{\mathbb{R}^D} \) is a pointwise vector norm (e.g.,
            Euclidean),
        - \( p \geq 1 \) is the order of the Lp norm.

    This formulation is particularly useful in applications where the
    importance of different time intervals varies. For example:
        - In energy forecasting, peak hours might be weighted more heavily.
        - In finance, more recent values may be considered more relevant.
        - In medicine, specific periods of a signal (e.g., during a symptom)
        can be prioritized.

    If no weight is specified, the distance reduces to the standard unweighted
    Lp distance.

    The distance supports `FDataGrid` and `FDataBasis` objects. The integration
    is performed using Simpson's rule for `FDataGrid` and multidimensional
    quadrature (`nquad_vec`) for `FDataBasis`.

    Args:
        p: Exponent of the Lp norm. Must be ≥ 1. If set to ``math.inf``, the
        distance becomes the L-infinity metric. Defaults to 2.
        vector_norm: Norm used pointwise for vector valued functions.
            If a float is passed, it is interpreted as the Lp norm index in
            \( \mathbb{R}^D \). If ``None``, defaults to the value of ``p``.
        lp_weight: Optional weight to apply during integration.
            Can be a float (uniform weight) or a callable \( w(t) \) that
            returns pointwise weights over the domain.

    Examples:
        Computes the weighted L2 distance between y = 1 and y = 0
        over the interval [0, 1], assigning double weight to every point.

        >>> import skfda
        >>> import numpy as np
        >>> from skfda.misc.metrics import WeightedLpDistance
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd1 = skfda.FDataGrid([np.ones(len(x))], x)
        >>> fd2 = skfda.FDataGrid([np.zeros(len(x))], x)
        >>>
        >>> distance = WeightedLpDistance(p=2, lp_weight=2.0)
        >>> distance(fd1, fd2).round(2)
        array([1.41])

        If the functional data are defined over different sets of
        discretization points, the function raises an exception.

        >>> x2 = np.linspace(0, 2, 1001)
        >>> fd3 = skfda.FDataGrid([np.zeros(len(x2))], x2)
        >>> distance(fd1, fd3)
        Traceback (most recent call last):
            ...
        ValueError: ...

    See Also:
        :class:`LpDistance`: Unweighted version of this distance.
        :class:`WeightedLpNorm`: Underlying norm used for distance computation.
    """

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
    """
    Functional wrapper for computing weighted Lp norms.

    See :class:`~skfda.misc.metrics.WeightedLpDistance` for full documentation.
    """
    return WeightedLpDistance(
        p=p,
        vector_norm=vector_norm,
        lp_weight=lp_weight,
    )(
        fdata1,
        fdata2,
    )
