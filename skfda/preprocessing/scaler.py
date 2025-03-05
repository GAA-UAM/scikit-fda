"""Functional Standard Scaler."""

from __future__ import annotations

import warnings

import scipy.integrate

from .._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FData, FDataGrid, FDataBasis
import numpy as np

from ..exploratory import stats
from typing import Any, Union, Optional
from ..typing._numpy import NDArrayFloat


class FunctionalStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize the functional data.
    """

    def __init__(  # noqa: WPS211
        self,
        with_mean: bool = True,
        with_std: bool = True,
        mean: Union[NDArrayFloat, None] = None,
        scale: Union[NDArrayFloat, None] = None,
        scaling_method: str = "functional",
        centering_method: str = "functional",
        correction: int = 0,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = mean
        self.scale_ = scale
        self.scaling_method_ = scaling_method
        self.centering_method_ = centering_method
        self.correction_ = correction

    def fit(self, X: FData, y: Optional[Any] = None) -> FunctionalStandardScaler:
        """Compute the mean and standard deviation for each functional component."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            raise TypeError("X must be an FDataGrid or FDataBasis object.")

        if self.scaling_method_ == "functional":
            self.mean_ = stats.mean(X) if self.with_mean else None
            self.scale_ = (
                stats.std(X, correction=self.correction_) if self.with_std else None
            )

        elif self.scaling_method_ == "scalar":
            if self.with_mean:
                if isinstance(X, FDataGrid):
                    self.mean_ = X.data_matrix.mean(axis=0).mean()
                elif isinstance(X, FDataBasis):
                    start, end = X.domain_range[0]
                    integral = scipy.integrate.quad_vec(
                        lambda x: X.mean()(x),
                        start,
                        end,
                    )
                    self.mean_ = integral[0].flatten() * 1 / (end - start)
            else:
                self.mean_ = None

            if self.with_std:
                if isinstance(X, FDataGrid):
                    mean = X.data_matrix.mean(axis=0).mean()
                    integrand = X.copy(
                        data_matrix=(X.data_matrix - mean) ** 2,
                        coordinate_names=(None,),
                    ).mean()
                    self.scale_ = np.sqrt(
                        integrand.integrate().ravel()
                        * X.n_samples
                        / (X.n_samples - self.correction_)
                        * 1
                        / (X.grid_points[1] - X.grid_points[0])
                    )

                elif isinstance(X, FDataBasis):
                    start, end = X.domain_range[0]
                    integral = scipy.integrate.quad_vec(
                        lambda x: X.mean()(x),
                        start,
                        end,
                    )
                    mean = integral[0].flatten() * 1 / (end - start)

                    integral = scipy.integrate.quad_vec(
                        lambda x: (X(x) - mean) ** 2,
                        start,
                        end,
                    )
                    self.scale_ = np.sqrt(
                        integral[0].flatten().sum()
                        * 1
                        / (end - start)
                        * 1
                        / (X.n_samples - self.correction_)
                    )
            else:
                self.scale_ = None
        else:
            raise ValueError("Invalid scaling method. Choose 'functional' or 'scalar'.")

        return self

    def transform(self, X: FData) -> FData:
        """Standardize the functional data based on the computed mean and scale."""
        if not isinstance(X, (FDataGrid, FDataBasis)):
            raise TypeError("X must be an FDataGrid or FDataBasis object.")

        if self.with_mean and self.mean_ is not None:
            X = X - self.mean_

        if self.with_std and self.scale_ is not None:
            X = X / self.scale_

        return X
