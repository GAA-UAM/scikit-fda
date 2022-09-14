
"""Implementation of Lp distances."""
from __future__ import annotations

import math
from typing import Optional, TypeVar, Union

import numpy as np
from typing_extensions import Final

from ...representation import FData
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat
from ._lp_norms import LpNorm
from ._utils import NormInducedMetric, pairwise_metric_optimization

T = TypeVar("T", NDArrayFloat, FData)


class LpDistance(NormInducedMetric[Union[NDArrayFloat, FData]]):
    r"""
    Lp distance for functional data objects.

    Calculates the distance between two functional objects.

    For each pair of observations f and g the distance between them is defined
    as:

    .. math::
        d(x, y) = \| x - y \|_p

    where :math:`\| {}\cdot{} \|_p` denotes the :func:`Lp norm <lp_norm>`.

    The objects ``l1_distance``, ``l2_distance`` and ``linf_distance`` are
    instances of this class with commonly used values of ``p``, namely 1, 2 and
    infinity.

    Args:
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x))], x)
        >>> fd2 =  skfda.FDataGrid([np.zeros(len(x))], x)
        >>>
        >>> distance = skfda.misc.metrics.LpDistance(p=2)
        >>> distance(fd, fd2).round(2)
        array([ 1.])


        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = np.linspace(0, 2, 1001)
        >>> fd2 = skfda.FDataGrid([np.zeros(len(x)), x/2 + 0.5], x)
        >>> distance = skfda.misc.metrics.LpDistance(p=2)
        >>> distance(fd, fd2)
        Traceback (most recent call last):
            ...
        ValueError: ...

    """  # noqa: P102

    def __init__(
        self,
        p: float,
        vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
    ) -> None:

        self.p = p
        self.vector_norm = vector_norm
        norm = LpNorm(p=p, vector_norm=vector_norm)

        super().__init__(norm)

    # This method is retyped here to work with either arrays or functions
    def __call__(self, elem1: T, elem2: T) -> NDArrayFloat:  # noqa: WPS612
        return super().__call__(elem1, elem2)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p={self.p}, vector_norm={self.vector_norm})"
        )


l1_distance: Final = LpDistance(p=1)
l2_distance: Final = LpDistance(p=2)
linf_distance: Final = LpDistance(p=math.inf)


@pairwise_metric_optimization.register
def _pairwise_metric_optimization_lp_fdata(
    metric: LpDistance,
    elem1: FData,
    elem2: Optional[FData],
) -> NDArrayFloat:
    from ...misc import inner_product, inner_product_matrix

    vector_norm = metric.vector_norm

    if vector_norm is None:
        vector_norm = metric.p

    # Special case, the inner product is heavily optimized
    if metric.p == vector_norm == 2:
        diag1 = inner_product(elem1, elem1)
        diag2 = diag1 if elem2 is None else inner_product(elem2, elem2)

        if elem2 is None:
            elem2 = elem1

        inner_matrix = inner_product_matrix(elem1, elem2)

        distance_matrix_sqr = (
            -2 * inner_matrix
            + diag1[:, np.newaxis]
            + diag2[np.newaxis, :]
        )

        np.clip(
            distance_matrix_sqr,
            a_min=0,
            a_max=None,
            out=distance_matrix_sqr,
        )

        return np.sqrt(distance_matrix_sqr)  # type: ignore[no-any-return]

    return NotImplemented


def lp_distance(
    fdata1: T,
    fdata2: T,
    *,
    p: float,
    vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
) -> NDArrayFloat:
    r"""
    Lp distance for FDataGrid objects.

    Calculates the distance between two functional objects.

    For each pair of observations f and g the distance between them is defined
    as:

    .. math::
        d(f, g) = d(g, f) = \| f - g \|_p

    where :math:`\| {}\cdot{} \|_p` denotes the :func:`Lp norm <lp_norm>`.

    Note:
        This function is a wrapper of :class:`LpDistance`, available only for
        convenience. As the parameter ``p`` is mandatory, it cannot be used
        where a fully-defined metric is required: use an instance of
        :class:`LpDistance` in those cases.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Returns:
        Numpy vector where the i-th coordinate has the distance between the
        i-th element of the first object and the i-th element of the second
        one.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array of size 2 with the
        computed l2 distance between the functions in the same position in
        both.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x)), x], x)
        >>> fd2 =  skfda.FDataGrid([np.zeros(len(x)), x/2], x)
        >>>
        >>> skfda.misc.metrics.lp_distance(fd, fd2, p=2).round(2)
        array([ 1.  ,  0.29])

        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = np.linspace(0, 2, 1001)
        >>> fd2 = skfda.FDataGrid([np.zeros(len(x)), x/2 + 0.5], x)
        >>> skfda.misc.metrics.lp_distance(fd, fd2, p=2)
        Traceback (most recent call last):
            ...
        ValueError: ...

    See also:
        :class:`~skfda.misc.metrics.LpDistance`

    """  # noqa: P102
    return LpDistance(p=p, vector_norm=vector_norm)(fdata1, fdata2)
