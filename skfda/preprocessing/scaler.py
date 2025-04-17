"""Functional Standard Scaler."""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatch
from typing import Any, TypeVar

import numpy as np

from skfda.representation import FData, FDataBasis, FDataGrid

from .._utils import function_to_fdatabasis
from .._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ..exploratory import stats
from ..typing._numpy import NDArrayFloat

T = TypeVar("T", bound=FDataGrid|FDataBasis)

""" 
def compute_uniform_center(X: FData) -> NDArrayFloat:
    Compute the uniform center of the functional data
    if isinstance(X, FDataGrid):
        mean = X.data_matrix.mean(axis=0).mean()
    elif isinstance(X, FDataBasis):
        arr = np.array(X.domain_range)
        diff = arr[:, 1] - arr[:, 0]
        integral = nquad_vec(
            lambda x: X.mean()(x),
            X.domain_range,
        )
        mean = integral / diff
    return mean """


@singledispatch
def center_scale(
    X: T,
    center: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
    scale: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
) -> FData:
    msg = f"center_scale not implemented for type {type(X)}"
    raise NotImplementedError(msg)


@center_scale.register
def _(
    X: FDataGrid,
    center: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
    scale: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
) -> FDataGrid:
    result = X.copy()

    if center is not None:
        center_val = center(X) if callable(center) else center
        if isinstance(center_val, FData):
            # Check is a FDataGrid with the same grid points
            result = result - center_val
        else:
            result.data_matrix -= np.asarray(center_val)

    if scale is not None:
        scale_val = scale(X) if callable(scale) else scale
        if isinstance(scale_val, FData):
            # Check is a FDataGrid with the same grid points
            result = result / scale_val
        else:
            result.data_matrix /= np.asarray(scale_val)

    return result


@center_scale.register
def _(
    X: FDataBasis,
    center: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
    scale: Callable[[FData], NDArrayFloat] | FData | NDArrayFloat | None,
) -> FDataBasis:
    result = X.copy()

    if center is not None:
        center_val = center(X) if callable(center) else center
        if isinstance(center_val, FData):
            # TODO: Check is a FDataBasis with the same grid points
            result = result - center_val
        else:
            result = function_to_fdatabasis(
                lambda x: (result(x) - np.asarray(center_val)),
                new_basis=result.basis,
            )

    if scale is not None:
        scale_val = scale(X) if callable(scale) else scale
        if isinstance(scale_val, FData):
            # TODO: Check is a FDataBasis with the same grid points
            result = result / scale_val
        else:
            result = function_to_fdatabasis(
                lambda x: (result(x) / np.asarray(scale_val)),
                new_basis=result.basis,
            )

    return result


class StandardScaler(
    BaseEstimator, InductiveTransformerMixin[T, T, Any],
):
    """
    Standardize functional data by centering and/or scaling.

    Supports both FDataGrid and FDataBasis representations.
    """

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        correction: int = 0,
    ) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.correction_ = correction

        self.mean_: FData | None = None
        self.scale_: FData | None = None

    def fit(
        self, X: T, y: Any | None = None,  # noqa: ANN401, ARG002
    ) -> StandardScaler:
        """Compute mean and standard deviation of the functional data."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            msg = "X must be an FDataGrid or FDataBasis object."
            raise TypeError(msg)

        self.mean_ = stats.mean(X)
        self.scale_ = stats.std(X, correction=self.correction_)
        return self

    def transform(self, X: T) -> T:
        """Standardize the functional data using the computed mean and scale."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            msg = "X must be an FDataGrid or FDataBasis object."
            raise TypeError(msg)
        if self.mean_ is None or self.scale_ is None:
            msg = "fit must be called before transform."
            raise ValueError(msg)

        if isinstance(X, FDataGrid):
            if self.with_mean:
                X = X - self.mean_
            if self.with_std:
                X = X / self.scale_

        elif isinstance(X, FDataBasis):
            mean_func = self.mean_ if self.with_mean else lambda x: 0
            if self.with_std:
                scale_func = self.scale_
                X = function_to_fdatabasis(
                    lambda x: (X(x) - mean_func(x)) / scale_func(x),
                    new_basis=X.basis,
                )
            elif self.with_mean:
                X = function_to_fdatabasis(
                    lambda x: X(x) - mean_func(x), new_basis=X.basis,
                )

        return X
