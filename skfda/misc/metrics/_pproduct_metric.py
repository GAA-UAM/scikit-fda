"""Implementation of Lp metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn, TypeVar

import multimethod
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat
from ..metrics._utils import pairwise_metric_optimization
from ..validation import check_fdata_same_kind
from ._utils import PairwiseMetric

V = TypeVar("V", bound=FData | pd.DataFrame | NDArrayFloat)


@multimethod.multidispatch
def compute_p_product(
    metric: PProductMetric[V],
    arg1: V,
    arg2: V,
) -> NoReturn:
    msg = (
        f"PProductMetric not implemented for type {type(arg1)} and "
        f"{type(arg2)}."
    )
    raise NotImplementedError(msg)


@compute_p_product.register
def _(
    metric: PProductMetric[V],
    arg1: NDArrayFloat,
    arg2: NDArrayFloat,
) -> NDArrayFloat:
    from ..metrics import l2_distance

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

    return (metric_computator(arg1, arg2) ** metric.p * weights) ** (
        1 / metric.p
    )


@compute_p_product.register
def _(metric: PProductMetric[V], arg1: FData, arg2: FData) -> NDArrayFloat:
    from ..metrics import l2_distance

    if not arg1.__eq__(arg2):
        msg = "FData objects must be equal to compute p-product."
        raise ValueError(msg)

    weights = metric.weights if metric.weights else 1.0
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
        data_matrix2 = arg1.data_matrix

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

        if isinstance(sample, (int, float, np.number, np.ndarray)):
            if v1.shape != v2.shape:
                msg = (
                    f"The shape of the column {col} must be "
                    f"the same for both DataFrames"
                )
                raise ValueError(msg)

        elif isinstance(sample, FData):
            check_fdata_same_kind(v1[0], v2[0])

        else:
            msg = f"Distance not supported for sample type {type(sample)} in column {col}"
            raise TypeError(msg)


@compute_p_product.register
def _(
    metric: PProductMetric[V],
    arg1: pd.DataFrame,
    arg2: pd.DataFrame,
) -> NDArrayFloat:

    same_structure_and_data(arg1, arg2)

    n_cols = arg1.shape[1]
    metrics = metric.metrics if metric.metrics else [default_metric] * n_cols
    weights = metric.weights if metric.weights else 1.0

    if isinstance(metrics, Metric):
        metrics = [metrics] * n_cols
    elif isinstance(metrics, Sequence):
        if len(metrics) != n_cols:
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
            fdata1 = FData._from_sequence(arg1[col])
            fdata2 = FData._from_sequence(arg2[col])
            distances[i,:] += metrics[i](fdata1, fdata2)
        else:
            distances[i,:] += metrics[i](arg1[col].values, arg2[col].values)

    res: NDArrayFloat = np.atleast_1d(
        np.sum(
            np.power(distances, metric.p) * weights[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        ),
    )
    return res[0] if len(res) == 1 else res

class DefaultMetric(Metric[V]):
    """Default metric based on the input type."""

    def __call__(
        self,
        arg1: NDArrayFloat | FData | pd.DataFrame,
        arg2: NDArrayFloat | FData | pd.DataFrame,
    ) -> NDArrayFloat:
        """Compute the distance between `arg1` and `arg2`."""
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            diff = arg1- arg2
            res = np.abs(diff).astype(np.float64)
            return res[0] if len(res) == 1 else res

        if isinstance(arg1, FData) and isinstance(arg2, FData):
            from skfda.misc.metrics import l2_distance
            return l2_distance(arg1, arg2)

        if isinstance(arg1, pd.DataFrame) and isinstance(arg2, pd.DataFrame):
            metric: PProductMetric[pd.DataFrame] = PProductMetric(p=2)
            return metric(arg1, arg2)

        msg = f"Unsupported types {type(arg1)} and {type(arg2)} for DefaultMetric."
        raise TypeError(msg)
def default_metric(
    arg1: NDArrayFloat | FData | pd.DataFrame,
    arg2: NDArrayFloat | FData | pd.DataFrame,
)-> NDArrayFloat:
    return DefaultMetric()(arg1=arg1, arg2=arg2)

class PProductMetric(Metric[V]):
    def __init__(
        self,
        p: float,
        metrics: (
            Sequence[Metric[V]] | Metric[V] | dict[str, Metric[V]] | None
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

    def __call__(self, arg1: V, arg2: V) -> NDArrayFloat:
        return compute_p_product(self, arg1, arg2)


def pproduct_metric(
    arg1: V,
    arg2: V,
    *,
    p: float,
    metrics: list[Metric[V]] | Metric[V] | dict[str, Metric[V]] | None = None,
    weights: NDArrayFloat | float | None = None,
) -> NDArrayFloat:
    metric = PProductMetric(p, metrics=metrics, weights=weights)
    return metric(arg1, arg2)


@pairwise_metric_optimization.register
def pairwise_metric_optimization_pproductmetric(
    metric: PProductMetric[V],
    arg1: pd.DataFrame,
    arg2: pd.DataFrame | None = None,
) -> NDArrayFloat:
    """Pairwise metric optimization for PProductMetric."""
    same_structure_and_data(arg1, arg2 if arg2 is not None else arg1)

    if arg2 is None:
        arg2 = arg1

    n_cols = arg1.shape[1]
    metrics = metric.metrics if metric.metrics else [default_metric] * n_cols
    weights = metric.weights if metric.weights else 1.0

    if isinstance(metrics, Metric):
        metrics = [metrics] * n_cols
    elif isinstance(metrics, Sequence):
        if len(metrics) != n_cols:
            msg = f"Number of metrics ({len(metrics)}) does not match number of columns ({n_cols})."
            raise ValueError(msg)
    elif isinstance(metrics, dict):
        if len(metrics) != n_cols:
            msg = f"Number of metrics ({len(metrics)}) does not match number of columns ({n_cols})."
            raise ValueError(msg)
        for col in metrics:
            if col not in arg1.columns:
                msg = f"Column '{col}' not found in DataFrames."
                raise ValueError(msg)
        metrics = [metrics[col] for col in arg1.columns]

    if isinstance(weights, (float, int)):
        weights = np.full(n_cols, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != n_cols:
        msg = f"Number of weights ({len(weights)}) does not match number of columns ({n_cols})."
        raise ValueError(msg)

    distances = np.zeros((len(arg1), len(arg2)), dtype=np.float64)

    for i, col in enumerate(arg1.columns):
        sample = arg1.iloc[0][col]

        if isinstance(sample, FData):
            fdata1 = FData._from_sequence(arg1[col])
            fdata2 = FData._from_sequence(arg2[col])
            col_distances = PairwiseMetric(metrics[i])(fdata1, fdata2)
        else:
            col_distances = PairwiseMetric(metrics[i])(arg1[col].values, arg2[col].values)

        distances += weights[i] * np.power(col_distances, metric.p)

    return np.power(distances, 1 / metric.p)


