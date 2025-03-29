"""Implementation of Lp norms."""

from __future__ import annotations

import functools
from typing import Dict, List, NoReturn, TypeVar, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat

V = TypeVar("V", bound=FData | pd.DataFrame | NDArrayFloat)


@functools.singledispatch
def compute_p_product(metric: PProductMetric, vector) -> NoReturn:  # noqa: ANN001
    msg = f"PProductMetric not implemented for type {type(vector)}"
    raise NotImplementedError(msg)


@compute_p_product.register
def _(metric: PProductMetric, vector: np.ndarray) -> NDArrayFloat:
    weights = metric.weights if metric.weights is not None else 1.0

    if isinstance(weights, (float, int)):
        vector = vector * weights

    return np.linalg.norm(vector, ord=metric.p, axis=-1)


@compute_p_product.register
def _(metric: PProductMetric, vector: FData) -> NDArrayFloat:
    from ..metrics import l2_norm

    weights = metric.weights if metric.weights else 1.0
    norms = metric.norms

    if isinstance(norms, dict):
        msg = "Dict norms not supported for FData. Use list instead."
        raise ValueError(msg)

    D = vector.dim_codomain

    norms = norms if norms else [l2_norm] * D
    if isinstance(norms, Norm):
        norms = [norms] * D
    elif isinstance(norms, list):
        if len(norms) != D:
            msg = f"Number of norms ({len(norms)}) does not match the number of dimensions ({D})."
            raise ValueError(msg)

    if isinstance(weights, (float, int)):
        weights = np.full(D, weights)
    elif isinstance(weights, np.ndarray):
        if len(weights) != D:
            msg = f"Number of weights ({len(weights)}) does not match the number of dimensions ({D})."
            raise ValueError(msg)

    if isinstance(vector, FDataBasis):
        if D != 1:
            msg = "FDataBasis must be 1-dimensional."
            raise ValueError(msg)
        value = norms[0](vector)
        res = np.sum(np.power(value, metric.p) * weights, axis=0)

    elif isinstance(vector, FDataGrid):
        data_matrix = vector.data_matrix
        values = np.array(
            [
                norms[i](
                    FDataGrid(
                        data_matrix=data_matrix[:, :, i], grid_points=vector.grid_points
                    )
                )
                for i in range(D)
            ]
        )
        res = np.sum(np.power(values, metric.p) * weights, axis=0)
    else:
        msg = f"FData subtype {type(vector)} not supported."
        raise NotImplementedError(msg)

    return res[0] if len(res) == 1 else res


@compute_p_product.register
def _(metric: PProductMetric, vector: pd.DataFrame) -> NDArrayFloat:
    from ..metrics import l2_norm

    n_cols = vector.shape[1]
    norms = metric.norms if metric.norms else [l2_norm] * n_cols
    weights = metric.weights if metric.weights else 1.0

    if isinstance(norms, Norm):
        norms = [norms] * n_cols
    elif isinstance(norms, list):
        if len(norms) != n_cols:
            msg = f"Number of norms ({len(norms)}) does not match the number of columns ({n_cols})."
            raise ValueError(msg)
    elif isinstance(norms, dict):
        if len(norms) != n_cols:
            msg = f"Number of norms ({len(norms)}) does not match the number of columns ({n_cols})."
            raise ValueError(msg)
        for col in norms.keys():
            if col not in vector.columns:
                msg = f"Column '{col}' not found in DataFrame."
                raise ValueError(msg)
        norms = [norms[col] for col in vector.columns]

    if isinstance(weights, (float, int)):
        weights = np.full(n_cols, weights)
    elif isinstance(weights, np.ndarray):
        if len(weights) != n_cols:
            msg = f"Number of weights ({len(weights)}) does not match the number of columns ({n_cols})."
            raise ValueError(msg)

    values = np.array([norms[i](vector.iloc[:, i]) for i in range(n_cols)])

    return np.sum(np.power(values, metric.p) * weights, axis=0)


class PProductMetric(Norm):
    def __init__(
        self,
        p: float,
        norms: list[Norm] | Norm | dict[str, Norm] | None = None,
        weights: NDArrayFloat | float | None = None,
    ) -> None:
        if not np.isinf(p) and p < 1:
            msg = f"p (={p}) must be equal or greater than 1."
            raise ValueError(msg)
        self.p = p
        self.norms = norms
        self.weights = weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, weights={self.weights})"

    def __call__(self, vector) -> NDArrayFloat:
        return compute_p_product(self, vector)


def pproduct_metric(
    vector: V,
    *,
    p: float,
    norms: list[Norm] | Norm | dict[str, Norm] | None = None,
    weights: NDArrayFloat | float | None = None,
) -> NDArrayFloat:
    metric = PProductMetric(p, norms=norms, weights=weights)
    return metric(vector)
