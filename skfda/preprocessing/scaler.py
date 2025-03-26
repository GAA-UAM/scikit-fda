"""Functional Standard Scaler."""

from __future__ import annotations

from .._utils import function_to_fdatabasis, nquad_vec
from .._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FData, FDataGrid, FDataBasis
import numpy as np

from ..exploratory import stats
from typing import Any, Optional, Callable
from ..typing._numpy import NDArrayFloat


def compute_uniform_center(X: FData) -> NDArrayFloat:
    """Compute the uniform center of the functional data."""
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
    return mean


def compute_uniform_scale(X: FData, correction: int) -> NDArrayFloat:
    """Compute the uniform scale of the functional data."""
    if isinstance(X, FDataGrid):
        mean = X.data_matrix.mean(axis=0).mean()
        integrand = X.copy(
            data_matrix=(X.data_matrix - mean) ** 2,
            coordinate_names=(None,),
        ).mean()
        scale = np.sqrt(
            integrand.integrate().ravel()
            * X.n_samples
            / (X.n_samples - correction)
            * 1
            / (X.grid_points[1] - X.grid_points[0])
        )

    elif isinstance(X, FDataBasis):

        mean = compute_uniform_center(X)

        arr = np.array(X.domain_range)
        diff = arr[:, 1] - arr[:, 0]

        integral = nquad_vec(
            lambda x: (X(x) - mean) ** 2,
            X.domain_range,
        )
        scale = np.sqrt(integral * 1 / (diff) * 1 / (X.n_samples - correction))
    return scale


def center_scale(
    X: FData,
    center: Optional[Callable[[NDArrayFloat], NDArrayFloat]],
    scale: Optional[Callable[[NDArrayFloat], NDArrayFloat]],
) -> FData:
    pass


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize functional data by centering and/or scaling.

    Supports both FDataGrid and FDataBasis representations.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        correction: int = 0,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.correction_ = correction

        self.mean_: Optional[FData] = None
        self.scale_: Optional[FData] = None

    def fit(self, X: FData, y: Optional[Any] = None) -> "StandardScaler":
        """Compute mean and standard deviation of the functional data."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            raise TypeError("X must be an FDataGrid or FDataBasis object.")

        self.mean_ = stats.mean(X)
        self.scale_ = stats.std(X, correction=self.correction_)
        return self

    def transform(self, X: FData) -> FData:
        """Standardize the functional data using the computed mean and scale."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            raise TypeError("X must be an FDataGrid or FDataBasis object.")
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("fit must be called before transform.")

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
                    lambda x: (X(x) - mean_func(x)) / scale_func(x), new_basis=X.basis
                )
            elif self.with_mean:
                X = function_to_fdatabasis(
                    lambda x: X(x) - mean_func(x), new_basis=X.basis
                )

        return X
