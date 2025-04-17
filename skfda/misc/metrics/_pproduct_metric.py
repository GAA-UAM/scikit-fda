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

V = TypeVar("V", bound=FData | pd.DataFrame | NDArrayFloat)


@multimethod.multidispatch
def compute_p_product(
    metric: PProductMetric[V],
    arg1: V,
    arg2: V,
) -> NoReturn:
    msg = f"PProductMetric not implemented for type {type(arg1)} and \
        {type(arg2)}."
    raise NotImplementedError(msg)


@compute_p_product.register
def _(
    metric: PProductMetric[V],
    arg1:  NDArrayFloat,
    arg2:  NDArrayFloat,
) -> NDArrayFloat:
    from ..metrics import l2_distance

    weights = metric.weights if metric.weights is not None else 1.0
    if not isinstance(weights, (float, int)):
        msg = f"Only float or int weights are supported. Got {type(weights)}."
        raise TypeError(msg)

    if arg1.shape != arg2.shape:
        msg = f"Shapes {arg1.shape} and {arg2.shape} do not match."
        raise ValueError(msg)

    metric_computator = (
        metric.metrics if metric.metrics else l2_distance
    )

    if not isinstance(metric_computator, Metric):
        msg = f"Only one metric is supported for NDArrayFloat. \
            Got {metric.metrics}."
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
        msg = f"Number of metrics ({len(metrics)}) does not match the number\
                of dimensions ({D})."
        raise ValueError(msg)

    if isinstance(weights, (float, int)):
        weights = np.full(D, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != D:
        msg = f"Number of weights ({len(weights)}) does not match the\
                number of dimensions ({D})."
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
        np.sum(np.power(value, metric.p) * weights, axis=0, dtype=np.float64)
    )
    return res[0] if len(res) == 1 else res


def same_structure_and_data(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return df1.dtypes.equals(df2.dtypes) and all(
        (
            (
                np.issubdtype(dtype, np.number)
                and df1[col].shape == df2[col].shape
            )
            or (issubclass(dtype.type, FData) and df1[col].__eq__(df2[col]))
            or not (
                np.issubdtype(dtype, np.number)
                or issubclass(dtype.type, FData)
            )
        )
        for col, dtype in df1.dtypes.items()
    )


@compute_p_product.register
def _(
    metric: PProductMetric[V],
    arg1: pd.DataFrame,
    arg2: pd.DataFrame,
) -> NDArrayFloat:
    from ..metrics import l2_distance

    if not same_structure_and_data(arg1, arg2):
        msg = "DataFrames must have the same structure and data to compute \
            p-product."
        raise ValueError(msg)

    n_cols = arg1.shape[1]
    metrics = metric.metrics if metric.metrics else [l2_distance] * n_cols
    weights = metric.weights if metric.weights else 1.0

    if isinstance(metrics, Metric):
        metrics = [metrics] * n_cols
    elif isinstance(metrics, Sequence):
        if len(metrics) != n_cols:
            msg = f"Number of metrics ({len(metrics)}) does not match the \
                number of columns ({n_cols})."
            raise ValueError(msg)
    elif isinstance(metrics, dict):
        if len(metrics) != n_cols:
            msg = f"Number of metrics ({len(metrics)}) does not match the \
                number of columns ({n_cols})."
            raise ValueError(msg)
        for col in metrics:
            if col not in arg1.columns:
                msg = f"Column '{col}' not found in DataFrames."
                raise ValueError(msg)
        metrics = [metrics[col] for col in arg1.columns]

    if isinstance(weights, (float, int)):
        weights = np.full(n_cols, weights)
    elif isinstance(weights, np.ndarray) and len(weights) != n_cols:
        msg = f"Number of weights ({len(weights)}) does not match the \
            number of columns ({n_cols})."
        raise ValueError(msg)

    values = np.array(
        [metrics[i](arg1.iloc[:, i], arg2.iloc[:, i]) for i in range(n_cols)],
    )
    res: NDArrayFloat = np.atleast_1d(
        np.sum(np.power(values, metric.p) * weights, axis=0, dtype=np.float64),
    )
    return res[0] if len(res) == 1 else res


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
