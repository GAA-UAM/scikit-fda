"""Utilities for norms and metrics."""
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

import multimethod
import numpy as np

from ..._utils import _MapAcceptable, _pairwise_symmetric
from ...representation import FData, FDataGrid
from ...typing._base import Vector
from ...typing._metric import Metric, Norm, VectorType
from ...typing._numpy import NDArrayFloat

_MapAcceptableT = TypeVar("_MapAcceptableT", bound=_MapAcceptable)
T = TypeVar("T", bound=FData)


def _check_compatible(fdata1: T, fdata2: T) -> None:

    if isinstance(fdata1, FData) and isinstance(fdata2, FData):
        if (
            fdata2.dim_codomain != fdata1.dim_codomain
            or fdata2.dim_domain != fdata1.dim_domain
        ):
            raise ValueError("Objects should have the same dimensions")

        if not np.array_equal(fdata1.domain_range, fdata2.domain_range):
            raise ValueError("Domain ranges for both objects must be equal")


def _cast_to_grid(
    fdata1: FData,
    fdata2: FData,
    eval_points: Optional[NDArrayFloat] = None,
    _check: bool = True,
) -> Tuple[FDataGrid, FDataGrid]:
    """
    Convert fdata1 and fdata2 to FDatagrid.

    Checks if the fdatas passed as argument are unidimensional and compatible
    and converts them to FDatagrid to compute their distances.

    Args:
        fdata1: First functional object.
        fdata2: Second functional object.
        eval_points: Evaluation points.

    Returns:
        Tuple with two :obj:`FDataGrid` with the same grid points.
    """
    # Dont perform any check
    if not _check:
        return fdata1, fdata2

    _check_compatible(fdata1, fdata2)

    # Case new evaluation points specified
    if eval_points is not None:  # noqa: WPS223
        fdata1 = fdata1.to_grid(eval_points)
        fdata2 = fdata2.to_grid(eval_points)

    elif not isinstance(fdata1, FDataGrid) and isinstance(fdata2, FDataGrid):
        fdata1 = fdata1.to_grid(fdata2.grid_points[0])

    elif not isinstance(fdata2, FDataGrid) and isinstance(fdata1, FDataGrid):
        fdata2 = fdata2.to_grid(fdata1.grid_points[0])

    elif (
        not isinstance(fdata1, FDataGrid)
        and not isinstance(fdata2, FDataGrid)
    ):
        domain = fdata1.domain_range[0]
        grid_points = np.linspace(*domain)
        fdata1 = fdata1.to_grid(grid_points)
        fdata2 = fdata2.to_grid(grid_points)

    elif not np.array_equal(
        fdata1.grid_points,
        fdata2.grid_points,
    ):
        raise ValueError(
            "Grid points for both objects must be equal or"
            "a new list evaluation points must be specified",
        )

    return fdata1, fdata2


class NormInducedMetric(Metric[VectorType]):
    r"""
    Metric induced by a norm.

    Given a norm :math:`\| \cdot \|: X \rightarrow \mathbb{R}`,
    returns the metric :math:`d: X \times X \rightarrow \mathbb{R}` induced
    by the norm:

    .. math::
        d(f,g) = \|f - g\|

    Args:
        norm: Norm used to induce the metric.

    Examples:
        Computes the :math:`\mathbb{L}^2` distance between an object containing
        functional data corresponding to the function :math:`y(x) = x` defined
        over the interval [0, 1] and another one containing data of the
        function :math:`y(x) = x/2`.

        Firstly we create the functional data.

        >>> import skfda
        >>> import numpy as np
        >>> from skfda.misc.metrics import l2_norm, NormInducedMetric
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([x], x)
        >>> fd2 = skfda.FDataGrid([x/2], x)

        To construct the :math:`\mathbb{L}^2` distance it is used the
        :math:`\mathbb{L}^2` norm wich it is used to compute the distance.

        >>> l2_distance = NormInducedMetric(l2_norm)
        >>> d = l2_distance(fd, fd2)
        >>> float('%.3f'% d)
        0.289

    """

    def __init__(self, norm: Norm[VectorType]):
        self.norm = norm

    def __call__(self, elem1: VectorType, elem2: VectorType) -> NDArrayFloat:
        """Compute the induced norm between two vectors."""
        return self.norm(elem1 - elem2)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(norm={self.norm})"


@multimethod.multidispatch
def pairwise_metric_optimization(
    metric: Any,
    elem1: Any,
    elem2: Optional[Any],
) -> NDArrayFloat:
    """
    Optimized computation of a pairwise metric.

    This is a generic function that can be subclassed for different
    combinations of metric and operators in order to provide a more
    efficient implementation for the pairwise metric matrix.
    """
    return NotImplemented


class PairwiseMetric(Generic[_MapAcceptableT]):
    """
    Pairwise metric function.

    Computes a given metric pairwise. The matrix returned by the pairwise
    metric is a matrix with as many rows as observations in the first object
    and as many columns as observations in the second one. Each element
    (i, j) of the matrix is the distance between the ith observation of the
    first object and the jth observation of the second one.

    Args:
        metric: Metric between two elements of a metric
            space.

    """

    def __init__(
        self,
        metric: Metric[_MapAcceptableT],
    ):
        self.metric = metric

    def __call__(
        self,
        elem1: _MapAcceptableT,
        elem2: Optional[_MapAcceptableT] = None,
    ) -> NDArrayFloat:
        """Evaluate the pairwise metric."""
        optimized = pairwise_metric_optimization(self.metric, elem1, elem2)

        return (
            _pairwise_symmetric(
                self.metric,  # type: ignore[arg-type]
                elem1,
                elem2,
            )
            if optimized is NotImplemented
            else optimized
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metric={self.metric})"


Original = TypeVar("Original", bound=Vector)
Transformed = TypeVar("Transformed", bound=Vector)


class TransformationMetric(Generic[Original, Transformed], Metric[Original]):
    """
    Compute a distance after transforming the data.

    This is a convenience function to compute a metric after a transformation
    is applied to the data. It can be used, for example, to compute
    Sobolev-like metrics.

    Args:
        e1: First object.
        e2: Second object.

    Returns:
        Distance.

    Examples:
        Compute the L2 distance between the function derivatives.

        >>> import skfda
        >>> from skfda.misc.metrics import l2_distance, TransformationMetric

        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([x], x)
        >>> fd2 = skfda.FDataGrid([x/2], x)

        >>> dist = TransformationMetric(
        ...     transformation=lambda x: x.derivative(),
        ...     metric=l2_distance,
        ... )
        >>> dist(fd, fd2)
        array([ 0.5])

    """

    def __init__(
        self,
        transformation: Callable[[Original], Transformed],
        metric: Metric[Transformed],
    ):
        self.transformation = transformation
        self.metric = metric

    def __call__(
        self,
        e1: Original,
        e2: Original,
    ) -> NDArrayFloat:
        """Compute the distance."""
        e1_trans = self.transformation(e1)
        e2_trans = self.transformation(e2)

        return self.metric(e1_trans, e2_trans)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}()"
        )


@pairwise_metric_optimization.register
def _pairwise_metric_optimization_transformation_dist(
    metric: TransformationMetric[Any, Any],
    e1: T,
    e2: Optional[T],
) -> NDArrayFloat:

    e1_trans = metric.transformation(e1)
    e2_trans = None if e2 is None else metric.transformation(e2)

    pairwise = PairwiseMetric(metric.metric)

    return pairwise(e1_trans, e2_trans)


def _fit_metric(metric: Metric[T], X: T) -> None:
    """Fits a metric if it has a fit method.

    Args:
        metric: The metric to fit.
        X: FData with the training data.
    """
    fit = getattr(metric, 'fit', lambda X: None)
    fit(X)
