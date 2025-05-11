"""Functional Standard Scaler."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from skfda.representation import FData, FDataBasis, FDataGrid

from .._utils import function_to_fdatabasis
from .._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ..exploratory import stats
from ..misc.validation import check_fdata_same_kind
from ..typing._numpy import NDArrayFloat

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=FDataGrid | FDataBasis)


class CenterScaler(BaseEstimator, InductiveTransformerMixin[T, T, Any]):
    """
    Transformer for centering and scaling functional data.

    This class applies centering and/or scaling to functional data objects
    (`FDataGrid` or `FDataBasis`). The centering and scaling parameters can
    either be predefined (constant or `FData` objects), or computed from the
    data using callable functions.

    Args:
        center : Centering transformation. If a callable, it will be applied to
            the input during `fit` to compute the center. If `None`, no
            centering is applied.
        scale : Scaling transformation. If a callable, it will be applied to
            the input during `fit` to compute the scale. If `None`, no
            scaling is applied.

    Attributes:
        center_ : Fitted centering value. Computed during `fit` if `center` is
            a callable.
        scale_ : Fitted scaling value. Computed during `fit` if `scale` is a
            callable.
    """

    def __init__(
        self,
        *,
        center: Callable[[T], NDArrayFloat | T] | T | NDArrayFloat | None,
        scale: Callable[[T], NDArrayFloat | T] | T | NDArrayFloat | None,
    ) -> None:
        self.center = center
        self.scale = scale

    def fit(self, X: T, y: Any | None = None) -> CenterScaler:  # noqa: ANN401, ARG002
        """
        Compute and store the centering and scaling parameters.

        Args:
            X : Functional data to compute parameters from.
            y : Present for compatibility with scikit-learn.

        Returns:
            self : Fitted transformer.
        """
        self.center_ = self.center(X) if callable(self.center) else self.center
        self.scale_ = self.scale(X) if callable(self.scale) else self.scale
        return self

    def transform(self, X: T) -> T:
        """
        Apply centering and scaling to the functional data.

        Args:
            X : Functional data to transform.

        Returns:
            X_new : Transformed functional data.
        """
        sklearn_check_is_fitted(self)
        return _transform(X, self.center_, self.scale_)


@singledispatch
def _transform(
    X: T,
    center: FData | NDArrayFloat | None,
    scale: FData | NDArrayFloat | None,
) -> FData:
    msg = f"transform not implemented for type {type(X)}"
    raise NotImplementedError(msg)


@_transform.register
def _transform_fdatagrid(
    X: FDataGrid,
    center: FData | NDArrayFloat | None,
    scale: FData | NDArrayFloat | None,
) -> FDataGrid:
    result = X.copy()

    if center is not None:
        if isinstance(center, FDataGrid):
            check_fdata_same_kind(X, center)
            if center.n_samples > 1:
                msg = "Cannot center with more than one sample"
                raise ValueError(msg)
            result = result - center
        else:
            result.data_matrix -= np.asarray(center)

    if scale is not None:
        if isinstance(scale, FDataGrid):
            check_fdata_same_kind(X, scale)
            if scale.n_samples > 1:
                msg = "Cannot scale with more than one sample"
                raise ValueError(msg)
            result = result / scale
        else:
            result.data_matrix /= np.asarray(scale)

    return result


@_transform.register
def _transform_fdatabasis(
    X: FDataBasis,
    center: FData | NDArrayFloat | None,
    scale: FData | NDArrayFloat | None,
) -> FDataBasis:
    result = X.copy()

    if center is not None:
        if isinstance(center, FDataBasis):
            check_fdata_same_kind(X, center)
            if center.n_samples > 1:
                msg = "Cannot center with more than one sample"
                raise ValueError(msg)
            result = result - center
        else:
            result = function_to_fdatabasis(
                lambda x: (result(x) - np.asarray(center)),
                new_basis=result.basis,
            )

    if scale is not None:
        if isinstance(scale, FDataBasis):
            check_fdata_same_kind(X, scale)
            if scale.n_samples > 1:
                msg = "Cannot scale with more than one sample"
                raise ValueError(msg)
            result = function_to_fdatabasis(
                lambda x: (result(x) / scale(x)),
                new_basis=result.basis,
            )
        else:
            result = function_to_fdatabasis(
                lambda x: (result(x) / np.asarray(scale)),
                new_basis=result.basis,
            )

    return result


class StandardScaler(
    CenterScaler,
):
    """
    Standardize functional data by centering and scaling.

    This transformer standardizes functional data by subtracting the functional
    mean and dividing by the standard deviation of the dataset. It supports
    both `FDataGrid` and `FDataBasis` representations.

    Args:
        with_mean : If True, center the data before scaling.
        with_std : If True, scale the data with the standard deviation.
        correction : Degrees of freedom correction to apply when computing the
            standard deviation.

    Attributes:
        mean_ : Mean function computed during `fit`. `None` if `with_mean` is
            False.
        scale_ : Standard deviation function computed during `fit`.
            `None` if `with_std` is False.
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
        self,
        X: T,
        y: Any | None = None,  # noqa: ANN401, ARG002
    ) -> StandardScaler:
        """
        Compute mean and standard deviation of the functional data.

        Args:
            X : Functional data to compute the statistics from.
            y : Not used, present for compatibility with scikit-learn pipeline.

        Returns:
            self : Fitted StandardScaler with computed `center_` and `scale_`.
        """
        self.center_ = None if not self.with_mean else stats.mean(X)
        self.scale_ = (
            None
            if not self.with_std
            else stats.std(X, correction=self.correction_)
        )
        return self
