"""Implementation of Product Metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, NoReturn, TypeVar

import multimethod
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..._utils._sklearn_adapter import (
    BaseEstimator,
)
from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat
from ..metrics._utils import pairwise_metric_optimization
from ..validation import check_fdata_same_kind
from ._utils import PairwiseMetric

V_call = TypeVar("V_call", bound=FData | pd.DataFrame | NDArrayFloat)
V_metric = TypeVar("V_metric", bound=FData | pd.DataFrame | NDArrayFloat)


@multimethod.multidispatch
def _compute_p_product(
    metric: PProductMetric[V_call, V_metric],
    arg1: V_call,
    arg2: V_call,
) -> NoReturn:
    msg = (
        f"PProductMetric not implemented for type {type(arg1)} and "
        f"{type(arg2)}."
    )
    raise NotImplementedError(msg)


@_compute_p_product.register
def _(
    metric: PProductMetric[V_call, V_metric],
    arg1: np.ndarray,
    arg2: np.ndarray,
) -> NDArrayFloat:
    from ..metrics import l2_distance  # noqa: PLC0415

    weights = metric.weights if metric.weights is not None else 1.0
    if not isinstance(weights, (float, int)):
        msg = f"Only float or int weights are supported. Got {type(weights)}."
        raise TypeError(msg)

    if arg1.shape != arg2.shape:
        msg = f"Shapes {arg1.shape} and {arg2.shape} do not match."
        raise ValueError(msg)

    metric_computator = metric.metrics if metric.metrics else l2_distance

    if not isinstance(metric_computator, Metric):
        msg = (
            f"Only one metric is supported for NDArrayFloat. "
            f"Got {metric.metrics}."
        )
        raise TypeError(msg)

    dist = metric_computator(arg1, arg2)
    return (weights * dist**metric.p) ** (1 / metric.p)


@_compute_p_product.register
def _(
    metric: PProductMetric[V_call, V_metric],
    arg1: FData,
    arg2: FData,
) -> NDArrayFloat:
    from ..metrics import l2_distance  # noqa: PLC0415

    weights = metric.weights if metric.weights is not None else 1.0
    metrics = metric.metrics

    if isinstance(metrics, dict):
        msg = "Dict metrics not supported for FData. Use list instead."
        raise TypeError(msg)

    D = arg1.dim_codomain  # noqa: N806

    metrics = metrics if metrics else [l2_distance] * D
    if isinstance(metrics, Metric):
        metrics = [metrics] * D
    elif isinstance(metrics, list) and len(metrics) != D:
        msg = (
            f"Number of metrics ({len(metrics)}) does not match the number"
            f" of dimensions ({D})."
        )
        raise ValueError(msg)

    if isinstance(weights, (float, int)):
        weights = np.full(D, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != D:
        msg = (
            f"Number of weights ({len(weights)}) does not match the"
            f" number of dimensions ({D})."
        )
        raise ValueError(msg)

    if isinstance(arg1, FDataBasis):
        if D != 1:
            msg = "FDataBasis must be 1-dimensional."
            raise ValueError(msg)
        value = metrics[0](arg1, arg2)

    elif isinstance(arg1, FDataGrid):
        data_matrix1 = arg1.data_matrix
        data_matrix2 = arg2.data_matrix

        value = np.array(
            [
                metrics[i](
                    FDataGrid(
                        data_matrix=data_matrix1[:, :, i],
                        grid_points=arg1.grid_points,
                    ),
                    FDataGrid(
                        data_matrix=data_matrix2[:, :, i],
                        grid_points=arg2.grid_points,
                    ),
                )
                for i in range(D)
            ],
        )
    else:
        msg = f"FData subtype {type(arg1)} not supported."
        raise NotImplementedError(msg)

    res: NDArrayFloat = np.atleast_1d(
        np.sum(np.power(value, metric.p) * weights, axis=0, dtype=np.float64),
    )
    return res[0] if len(res) == 1 else res


def same_structure_and_data(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    if not df1.columns.equals(df2.columns):
        msg = "Columns must be the same in both DataFrames"
        raise ValueError(msg)

    for col in df1.columns:
        v1, v2 = df1[col].values, df2[col].values  # noqa: PD011
        sample = df1.iloc[0][col]

        if isinstance(sample, FData):
            check_fdata_same_kind(v1[0], v2[0])

        elif not isinstance(sample, (int, float, np.number, np.ndarray)):
            msg = (
                f"Distance not supported for sample"
                f" type {type(sample)} in column {col}"
            )
            raise TypeError(msg)


@_compute_p_product.register
def _(  # noqa: C901, PLR0912
    metric: PProductMetric[V_call, V_metric],
    arg1: pd.DataFrame,
    arg2: pd.DataFrame,
) -> NDArrayFloat:
    same_structure_and_data(arg1, arg2)

    n_cols = arg1.shape[1]
    metrics = (
        metric.metrics
        if metric.metrics is not None
        else [default_metric] * n_cols
    )
    weights = metric.weights if metric.weights is not None else 1.0

    if isinstance(metrics, Metric):
        metrics = [metrics] * n_cols
    elif isinstance(metrics, Sequence) and len(metrics) != n_cols:
        msg = (
            f"Number of metrics ({len(metrics)}) does not match the"
            f" number of columns ({n_cols})."
        )
        raise ValueError(msg)
    elif isinstance(metrics, dict):
        if len(metrics) != n_cols:
            msg = (
                f"Number of metrics ({len(metrics)}) does not match the"
                f" number of columns ({n_cols})."
            )
            raise ValueError(msg)
        for col in metrics:
            if col not in arg1.columns:
                msg = f"Column '{col}' not found in DataFrames."
                raise ValueError(msg)
        metrics = [metrics[col] for col in arg1.columns]

    if isinstance(weights, (float, int)):
        weights = np.full(n_cols, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != n_cols:
        msg = (
            f"Number of weights ({len(weights)}) does not match the"
            f" number of columns ({n_cols})."
        )
        raise ValueError(msg)

    distances = np.zeros((len(arg1.columns), len(arg1)))

    for i, col in enumerate(arg1.columns):
        sample = arg1.iloc[0][col]
        if isinstance(sample, FData):
            fdata1 = FData._from_sequence(arg1[col])  # noqa: SLF001
            fdata2 = FData._from_sequence(arg2[col])  # noqa: SLF001
            distances[i, :] += metrics[i](fdata1, fdata2)
        else:
            distances[i, :] += metrics[i](arg1[col].values, arg2[col].values)
    if np.isinf(metric.p):
        res: NDArrayFloat = np.max(distances * weights[:, np.newaxis], axis=0)
    else:
        res = np.sum(
            np.power(distances, metric.p) * weights[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        res = np.power(res, 1 / metric.p)

    return res[0] if len(res) == 1 else res


class DefaultMetric(Metric[V_metric]):
    """
    Default metric class that computes distances based on the input data type.

    This class selects a distance computation method depending on the type of
    the input objects. It supports the following types:

    - ``np.ndarray``: Computes the element-wise absolute difference.
    - ``FData``: Uses the L2 distance for functional data objects.
    - ``pandas.DataFrame``: Applies a product metric with ``p=2`` across
    columns.

    If the input types are unsupported or mismatched, a ``TypeError`` is
    raised.

    """

    def __call__(
        self,
        arg1: NDArrayFloat | FData | pd.DataFrame,
        arg2: NDArrayFloat | FData | pd.DataFrame,
    ) -> NDArrayFloat:
        """
        Compute the distance between ``arg1`` and ``arg2``.

        The computation method depends on the type of the arguments:

        - If both are ``np.ndarray``, returns the absolute element-wise
        difference.
        - If both are ``FData``, returns the L2 distance.
        - If both are ``pandas.DataFrame``, uses a product metric with ``p=2``.

        Args:
            arg1 : NDArrayFloat or FData or pandas.DataFrame
                First object to compare. Must be of the same type as ``arg2``.
            arg2 : NDArrayFloat or FData or pandas.DataFrame
                Second object to compare. Must be of the same type as ``arg1``.

        Returns:
            The computed distance(s). The format depends on the input type:
            - For NumPy arrays, an array of absolute differences.
            - For FData, a float representing the L2 distance.
            - For DataFrames, a float from the product metric.
        """
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            diff = arg1 - arg2
            res = np.abs(diff).astype(np.float64)
            return res[0] if len(res) == 1 else res

        if isinstance(arg1, FData) and isinstance(arg2, FData):
            from skfda.misc.metrics import l2_distance  # noqa: PLC0415

            return l2_distance(arg1, arg2)

        if isinstance(arg1, pd.DataFrame) and isinstance(arg2, pd.DataFrame):
            metric: PProductMetric[pd.DataFrame, V_metric] = PProductMetric(
                p=2,
            )
            return metric(arg1, arg2)

        msg = (
            f"Unsupported types {type(arg1)} "
            f"and {type(arg2)} for DefaultMetric."
        )
        raise TypeError(msg)


def default_metric(
    arg1: NDArrayFloat | FData | pd.DataFrame,
    arg2: NDArrayFloat | FData | pd.DataFrame,
) -> NDArrayFloat:
    """
    Functional wrapper for computing Default Metric.

    See :class:`~skfda.misc.metrics.DefaultMetric` for full documentation.
    """
    return DefaultMetric()(arg1=arg1, arg2=arg2)


class PProductMetric(BaseEstimator, Metric[V_call], Generic[V_call, V_metric]):
    r"""
    Weighted :math:`l^p`-type product metric for Mixed Data.

    This class defines a generalized distance over product spaces, where each
    component of an observation may be of a different type (e.g., scalar,
    vector valued function, functional). The distance is computed as a weighted
    :math:`l^p` norm of component-wise distances, each using a custom metric.

    Given two observations
    :math:`X_1 = (X_1^{(1)}, ..., X_1^{(D)})` and
    :math:`X_2 = (X_2^{(1)}, ..., X_2^{(D)})`,
    the distance is computed as:

    .. math::
        d(X_1, X_2) = (\sum_{d=1}^D w^{(d)} * d_{q_d}(X_1^{(d)}, X_2^{(d)})^p
        )^{1/p}

    where:
        - :math:`d_{q_d}` is a metric defined for component `d` (e.g.,
            :math:`L^2` norm, Euclidean),
        - :math:`w^{(d)}` is a weight controlling the scale or relevance of
            component `d`,
        - :math:`p \in [1, \infinity]` controls how distances are aggregated.

    Args:
        p : Aggregation parameter. Must be >= 1 or `np.inf`.
            Determines the type of :math:`l^p` norm used to combine the
            component-wise distances.
        metrics : The metric(s) to use for each component. Options:
                - A single `Metric` object, used for all components.
                - A sequence of `Metric`, one per component.
                - A dictionary mapping keys to metrics, for use
                    with Mixed Data represented as `pd.DataFrames`.
                - `None` defaults to Euclidean for numeric values and
                    :math:`L^2` norm for `FData` types.
        weights : Weights for each component. If a scalar, all components
            receive equal weight. If `None`, weights default to 1 for all
            components.

    Examples:
    Calculates de product metric between two Mixed Data observations, each
    composed by a FDataBasis and a scalar value.

    >>> from skfda.misc.metrics import l2_distance
    >>> from skfda.representation.basis import FDataBasis, FourierBasis
    >>> from skfda.misc.metrics import PProductMetric
    >>> import numpy as np

    >>> basis = FourierBasis(n_basis=5)
    >>> fd1 = FDataBasis(basis, [[1, 2, 3, 4, 5]])
    >>> fd2 = FDataBasis(basis, [[2, 3, 4, 5, 6]])

    >>> pmetric = PProductMetric(
    ...     p=2,
    ...     metrics=[l2_distance, lambda x, y: abs(x - y)],
    ...     weights=np.array([1.0, 0.5]),
    ... )

    >>> import pandas as pd
    >>> df1 = pd.DataFrame({
    ...     "fd": [fd1],
    ...     "num": [6.0],
    ... })
    >>> df2 = pd.DataFrame({
    ...     "fd": [fd2],
    ...     "num": [3.0],
    ... })
    >>> pmetric(df1, df2).round(2)
    np.float64(3.08)

    Notes:
    - The metric is evaluated in a vectorized fashion. If `arg1` or `arg2`
      contains multiple observations, broadcasting rules apply.
    - This class enables integration of scalar, categorical (with encoding),
        and functional data for unified treatment in clustering,
        classification, etc.
    """

    def __init__(
        self,
        p: float,
        metrics: (
            Sequence[Metric[V_metric]]
            | Metric[V_metric]
            | dict[str, Metric[V_metric]]
            | None
        ) = None,
        weights: NDArrayFloat | float | None = None,
    ) -> None:
        if not np.isinf(p) and p < 1:
            msg = f"p (={p}) must be equal or greater than 1."
            raise ValueError(msg)
        self.p = p
        self.metrics = metrics
        self.weights = weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, weights={self.weights})"

    def __call__(self, arg1: V_call, arg2: V_call) -> NDArrayFloat:
        return _compute_p_product(self, arg1, arg2)


def pproduct_metric(
    arg1: V_call,
    arg2: V_call,
    *,
    p: float,
    metrics: (
        list[Metric[V_metric]]
        | Metric[V_metric]
        | dict[str, Metric[V_metric]]
        | None
    ) = None,
    weights: NDArrayFloat | float | None = None,
) -> NDArrayFloat:
    """
    Functional wrapper for computing PProduct Metrics.

    See :class:`~skfda.misc.metrics.PProductMetric` for full documentation.
    """
    metric: PProductMetric[V_call, V_metric] = PProductMetric(
        p,
        metrics=metrics,
        weights=weights,
    )
    return metric(arg1, arg2)


@pairwise_metric_optimization.register
def pairwise_metric_optimization_pproductmetric(  # noqa: C901, PLR0912
    metric: PProductMetric[V_call, V_metric],
    arg1: pd.DataFrame,
    arg2: pd.DataFrame | None = None,
) -> NDArrayFloat:
    """Pairwise metric optimization for PProductMetric."""
    same_structure_and_data(arg1, arg2 if arg2 is not None else arg1)

    if arg2 is None:
        arg2 = arg1

    n_cols = arg1.shape[1]
    metrics = (
        metric.metrics
        if metric.metrics is not None
        else [default_metric] * n_cols
    )
    weights = metric.weights if metric.weights is not None else 1.0

    if isinstance(metrics, Metric):
        metrics = [metrics] * n_cols
    elif isinstance(metrics, Sequence):
        if len(metrics) != n_cols:
            msg = (
                f"Number of metrics ({len(metrics)}) "
                f"does not match number of columns ({n_cols})."
            )
            raise ValueError(msg)
    elif isinstance(metrics, dict):
        if len(metrics) != n_cols:
            msg = (
                f"Number of metrics ({len(metrics)}) "
                f"does not match number of columns ({n_cols})."
            )
            raise ValueError(msg)
        for col in metrics:
            if col not in arg1.columns:
                msg = f"Column '{col}' not found in DataFrames."
                raise ValueError(msg)
        metrics = [metrics[col] for col in arg1.columns]

    if isinstance(weights, (float, int)):
        weights = np.full(n_cols, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != n_cols:
        msg = (
            f"Number of weights ({len(weights)}) "
            f"does not match number of columns ({n_cols})."
        )
        raise ValueError(msg)

    distances = np.zeros((len(arg1), len(arg2)), dtype=np.float64)

    for i, col in enumerate(arg1.columns):
        sample = arg1.iloc[0][col]

        if isinstance(sample, FData):
            fdata1 = FData._from_sequence(arg1[col])  # noqa: SLF001
            fdata2 = FData._from_sequence(arg2[col])  # noqa: SLF001
            col_distances = PairwiseMetric(metrics[i])(fdata1, fdata2)
        else:
            col_distances = PairwiseMetric(metrics[i])(
                arg1[col].values,
                arg2[col].values,
            )

        distances += weights[i] * np.power(col_distances, metric.p)

    return np.power(distances, 1 / metric.p)
