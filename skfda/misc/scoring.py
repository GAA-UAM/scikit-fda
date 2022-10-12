"""Scoring methods for FData."""
from __future__ import annotations

import math
import warnings
from functools import singledispatch
from typing import Callable, Optional, TypeVar, Union, overload

import numpy as np
import sklearn.metrics
from typing_extensions import Literal, Protocol

from .._utils import nquad_vec
from ..representation import FData, FDataBasis, FDataGrid
from ..representation._functional_data import EvalPointsType
from ..typing._numpy import NDArrayFloat

DataType = TypeVar('DataType')

MultiOutputType = Literal['uniform_average', 'raw_values']


class _InfiniteScoreError(Exception):
    """Exception for skipping integral on infinite value."""


class ScoreFunction(Protocol):
    """Type definition for score functions."""

    @overload
    def __call__(
        self,
        y_true: DataType,
        y_pred: DataType,
        *,
        sample_weight: NDArrayFloat | None = None,
        multioutput: Literal['uniform_average'] = 'uniform_average',
    ) -> float:
        pass  # noqa: WPS428

    @overload
    def __call__(   # noqa: D102
        self,
        y_true: DataType,
        y_pred: DataType,
        *,
        sample_weight: NDArrayFloat | None = None,
        multioutput: Literal['raw_values'],
    ) -> DataType:
        pass  # noqa: WPS428

    def __call__(   # noqa: D102
        self,
        y_true: DataType,
        y_pred: DataType,
        *,
        sample_weight: NDArrayFloat | None = None,
        multioutput: MultiOutputType = 'uniform_average',
    ) -> float | DataType:
        pass  # noqa: WPS428


def _domain_measure(fd: FData) -> float:
    measure = 1.0
    for interval in fd.domain_range:
        measure = measure * (interval[1] - interval[0])
    return measure


def _var(
    x: FDataGrid,
    weights: NDArrayFloat | None = None,
) -> FDataGrid:
    from ..exploratory.stats import mean, var

    if weights is None:
        return var(x)

    return mean(  # type: ignore[no-any-return]
        np.power(x - mean(x, weights=weights), 2),
        weights=weights,
    )


def _multioutput_score_basis(
    y_true: FDataBasis,
    multioutput: MultiOutputType,
    integrand: Callable[[NDArrayFloat], NDArrayFloat],
) -> float:

    if multioutput != "uniform_average":
        raise ValueError(
            f"Only \"uniform_average\" is supported for \"multioutput\" when "
            f"the input is a FDatabasis: received {multioutput} instead",
        )

    try:
        integral = nquad_vec(
            integrand,
            y_true.domain_range,
        )
    except _InfiniteScoreError:
        return -math.inf

    # If the dimension of the codomain is > 1,
    # the mean of the scores is taken
    return float(np.mean(integral) / _domain_measure(y_true))


def _multioutput_score_grid(
    score: FDataGrid,
    multioutput: MultiOutputType,
    squared: bool = True,
) -> float | FDataGrid:

    if not squared:
        score = np.sqrt(score)

    if multioutput == 'raw_values':
        return score

    # Score only contains 1 function
    # If the dimension of the codomain is > 1,
    # the mean of the scores is taken
    return float(np.mean(score.integrate()[0]) / _domain_measure(score))


@overload
def explained_variance_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
) -> float:
    pass  # noqa: WPS428


@overload
def explained_variance_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def explained_variance_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float | DataType:
    r"""Explained variance score for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the score is
    calculated as

    .. math::
        EV(y\_true, y\_pred)(t) = 1 -
        \frac{Var(y\_true(t) - y\_pred(t), sample\_weight)}
        {Var(y\_true(t), sample\_weight)}

    where :math:`Var` is a weighted variance.

    Weighted variance is defined as below

    .. math::
        Var(y\_true, sample\_weight)(t) = \sum_{i=1}^n w_i
        (X_i(t) - Mean(fd(t), sample\_weight))^2.

    Here, :math:`Mean` is a weighted mean.

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`EV` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`EV` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`EV` is
    calculated:

    .. math::
        mean(EV) = \frac{1}{V}\int_{D} EV(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    The best possible score is 1.0, lower values are worse.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.

    Returns:
        Explained variance score.

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return (  # type: ignore [no-any-return]
        sklearn.metrics.explained_variance_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
    )


@explained_variance_score.register  # type: ignore[attr-defined, misc]
def _explained_variance_score_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> Union[float, FDataGrid]:

    num = _var(y_true - y_pred, weights=sample_weight)
    den = _var(y_true, weights=sample_weight)

    # Divisions by zero allowed
    with np.errstate(divide='ignore', invalid='ignore'):
        score = 1 - num / den

    # 0 / 0 divisions should be 0 in this context, and the score, 1
    score.data_matrix[np.isnan(score.data_matrix)] = 1

    return _multioutput_score_grid(score, multioutput)


@explained_variance_score.register  # type: ignore[attr-defined, misc]
def _explained_variance_score_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    def _ev_func(x: EvalPointsType) -> NDArrayFloat:  # noqa: WPS430
        num = np.average(
            (
                (y_true(x) - y_pred(x))
                - np.average(
                    y_true(x) - y_pred(x),
                    weights=sample_weight,
                    axis=0,
                )
            ) ** 2,
            weights=sample_weight,
            axis=0,
        )

        den = np.average(
            (
                y_true(x)
                - np.average(y_true(x), weights=sample_weight, axis=0)
            ) ** 2,
            weights=sample_weight,
            axis=0,
        )

        # Divisions by zero allowed
        with np.errstate(divide='ignore', invalid='ignore'):
            score = 1 - num / den

        # 0/0 case, the score is 1.
        score[np.isnan(score)] = 1

        # r/0 case, r!= 0. Return -inf outside this function
        if np.any(np.isinf(score)):
            raise _InfiniteScoreError

        # Score only contains 1 input point
        assert score.shape[0] == 1
        return score[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _ev_func)


@overload
def mean_absolute_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
) -> float:
    pass  # noqa: WPS428


@overload
def mean_absolute_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def mean_absolute_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float | DataType:
    r"""Mean Absolute Error for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the error is
    calculated as

    .. math::
        MAE(y\_true, y\_pred)(t) = \frac{1}{\sum w_i}
        \sum_{i=1}^n w_i|X_i(t) - \hat{X}_i(t)|

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`MAE` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`MAE` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`MAE` is
    calculated:

    .. math::
        mean(MAE) = \frac{1}{V}\int_{D} MAE(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.

    Returns:
        Mean absolute error.

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return sklearn.metrics.mean_absolute_error(  # type: ignore [no-any-return]
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@mean_absolute_error.register  # type: ignore[attr-defined, misc]
def _mean_absolute_error_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> Union[float, FDataGrid]:
    from ..exploratory.stats import mean

    error = mean(np.abs(y_true - y_pred), weights=sample_weight)
    return _multioutput_score_grid(error, multioutput)


@mean_absolute_error.register  # type: ignore[attr-defined, misc]
def _mean_absolute_error_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    def _mae_func(x: EvalPointsType) -> NDArrayFloat:  # noqa: WPS430
        error = np.average(
            np.abs(y_true(x) - y_pred(x)),
            weights=sample_weight,
            axis=0,
        )

        # Error only contains 1 input point
        assert error.shape[0] == 1
        return error[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _mae_func)


@overload
def mean_absolute_percentage_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
) -> float:
    pass  # noqa: WPS428


@overload
def mean_absolute_percentage_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def mean_absolute_percentage_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float | DataType:
    r"""Mean Absolute Percentage Error for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the error is
    calculated as

    .. math::
        MAPE(y\_true, y\_pred)(t) = \frac{1}{\sum w_i}
        \sum_{i=1}^n w_i\frac{|X_i(t) - \hat{X}_i(t)|}{|X_i(t)|}

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`MAPE` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`MAPE` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`MAPE` is
    calculated:

    .. math::
        mean(MAPE) = \frac{1}{V}\int_{D} MAPE(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    This function should not be used if for some :math:`t` and some :math:`i`,
    :math:`X_i(t) = 0`.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.

    Returns:
        Mean absolute percentage error.

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return (  # type: ignore [no-any-return]
        sklearn.metrics.mean_absolute_percentage_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
    )


@mean_absolute_percentage_error.register  # type: ignore[attr-defined, misc]
def _mean_absolute_percentage_error_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> Union[float, FDataGrid]:
    from ..exploratory.stats import mean

    epsilon = np.finfo(np.float64).eps

    if np.any(np.abs(y_true.data_matrix) < epsilon):
        warnings.warn('Zero denominator', RuntimeWarning)

    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)

    error = mean(mape, weights=sample_weight)
    return _multioutput_score_grid(error, multioutput)


@mean_absolute_percentage_error.register  # type: ignore[attr-defined, misc]
def _mean_absolute_percentage_error_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    def _mape_func(x: EvalPointsType) -> NDArrayFloat:  # noqa: WPS430

        epsilon = np.finfo(np.float64).eps
        if np.any(np.abs(y_true(x)) < epsilon):
            warnings.warn('Zero denominator', RuntimeWarning)

        error = np.average(
            (
                np.abs(y_true(x) - y_pred(x))
                / np.maximum(np.abs(y_true(x)), epsilon)
            ),
            weights=sample_weight,
            axis=0,
        )

        # Error only contains 1 input point
        assert error.shape[0] == 1
        return error[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _mape_func)


@overload
def mean_squared_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
    squared: bool = True,
) -> float:
    pass  # noqa: WPS428


@overload
def mean_squared_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
    squared: bool = True,
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def mean_squared_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
    squared: bool = True,
) -> float | DataType:
    r"""Mean Squared Error for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the error is
    calculated as

    .. math::
        MSE(y\_true, y\_pred)(t) = \frac{1}{\sum w_i}
        \sum_{i=1}^n w_i(X_i(t) - \hat{X}_i(t))^2

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`MSE` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`MSE` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`MSE` is
    calculated:

    .. math::
        mean(MSE) = \frac{1}{V}\int_{D} MSE(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.
        squared: If True returns MSE value, if False returns RMSE value.

    Returns:
        Mean squared error.

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return sklearn.metrics.mean_squared_error(  # type: ignore [no-any-return]
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


@mean_squared_error.register  # type: ignore[attr-defined, misc]
def _mean_squared_error_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
    squared: bool = True,
) -> Union[float, FDataGrid]:
    from ..exploratory.stats import mean

    error: FDataGrid = mean(
        np.power(y_true - y_pred, 2),
        weights=sample_weight,
    )

    return _multioutput_score_grid(error, multioutput, squared=squared)


@mean_squared_error.register  # type: ignore[attr-defined, misc]
def _mean_squared_error_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    squared: bool = True,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    def _mse_func(x: EvalPointsType) -> NDArrayFloat:  # noqa: WPS430

        error: NDArrayFloat = np.average(
            (y_true(x) - y_pred(x)) ** 2,
            weights=sample_weight,
            axis=0,
        )

        if not squared:
            return np.sqrt(error)

        # Error only contains 1 input point
        assert error.shape[0] == 1
        return error[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _mse_func)


@overload
def mean_squared_log_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
    squared: bool = True,
) -> float:
    pass  # noqa: WPS428


@overload
def mean_squared_log_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
    squared: bool = True,
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def mean_squared_log_error(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
    squared: bool = True,
) -> float | DataType:
    r"""Mean Squared Log Error for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the error is
    calculated as

    .. math::
        MSLE(y\_true, y\_pred)(t) = \frac{1}{\sum w_i}
        \sum_{i=1}^n w_i(\log(1 + X_i(t)) - \log(1 + \hat{X}_i(t)))^2

    where :math:`\log` is the natural logarithm.

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`MSLE` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`MSLE` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`MSLE` is
    calculated:

    .. math::
        mean(MSLE) = \frac{1}{V}\int_{D} MSLE(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    This function should not be used if for some :math:`t` and some :math:`i`,
    :math:`X_i(t) < 0`.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.
        squared: default True. If False, square root is taken.

    Returns:
        Mean squared log error.

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return (  # type: ignore [no-any-return]
        sklearn.metrics.mean_squared_log_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=squared,
        )
    )


@mean_squared_log_error.register  # type: ignore[attr-defined, misc]
def _mean_squared_log_error_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
    squared: bool = True,
) -> Union[float, FDataGrid]:

    if np.any(y_true.data_matrix < 0) or np.any(y_pred.data_matrix < 0):
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets functions have negative values.",
        )

    return mean_squared_error(
        np.log1p(y_true),
        np.log1p(y_pred),
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


@mean_squared_log_error.register  # type: ignore[attr-defined, misc]
def _mean_squared_log_error_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    squared: bool = True,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    def _msle_func(x: EvalPointsType) -> NDArrayFloat:  # noqa: WPS430

        y_true_eval = y_true(x)
        y_pred_eval = y_pred(x)

        if np.any(y_true_eval < 0) or np.any(y_pred_eval < 0):
            raise ValueError(
                "Mean Squared Logarithmic Error cannot be used when "
                "targets functions have negative values.",
            )

        error: NDArrayFloat = np.average(
            (np.log1p(y_true_eval) - np.log1p(y_pred_eval)) ** 2,
            weights=sample_weight,
            axis=0,
        )

        if not squared:
            return np.sqrt(error)

        # Error only contains 1 input point
        assert error.shape[0] == 1
        return error[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _msle_func)


@overload
def r2_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['uniform_average'] = 'uniform_average',
) -> float:
    pass  # noqa: WPS428


@overload
def r2_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: Literal['raw_values'],
) -> DataType:
    pass  # noqa: WPS428


@singledispatch
def r2_score(
    y_true: DataType,
    y_pred: DataType,
    *,
    sample_weight: NDArrayFloat | None = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float | DataType:
    r"""R^2 score for :class:`~skfda.representation.FData`.

    With :math:`y\_true = (X_1, X_2, ..., X_n)` being the real values,
    :math:`t\_pred = (\hat{X}_1, \hat{X}_2, ..., \hat{X}_n)` being the
    estimated and :math:`sample\_weight = (w_1, w_2, ..., w_n)`, the score is
    calculated as

    .. math::
        R^2(y\_true, y\_pred)(t) = 1 -
        \frac{\sum_{i=1}^n w_i (X_i(t) - \hat{X}_i(t))^2}
        {\sum_{i=1}^n w_i (X_i(t) - Mean(y\_true, sample\_weight)(t))^2}

    where :math:`Mean` is a weighted mean.

    For :math:`y\_true` and :math:`y\_pred` of type
    :class:`~skfda.representation.FDataGrid`,  :math:`R^2` is
    also a :class:`~skfda.representation.FDataGrid` object with
    the same grid points.

    If multioutput = 'raw_values', the function :math:`R^2` is returned.
    Otherwise, if multioutput = 'uniform_average', the mean of :math:`R^2` is
    calculated:

    .. math::
        mean(R^2) = \frac{1}{V}\int_{D} R^2(t) dt

    where :math:`D` is the function domain and :math:`V` the volume of that
    domain.

    For :class:`~skfda.representation.FDataBasis` only
    'uniform_average' is available.

    If :math:`y\_true` and :math:`y\_pred` are numpy arrays, sklearn function
    is called.

    Args:
        y_true: Correct target values.
        y_pred: Estimated values.
        sample_weight: Sample weights. By default, uniform weights
            are taken.
        multioutput: Defines format of the return.

    Returns:
        R2 score

        If multioutput = 'uniform_average' or
        :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataBasis` objects, float is returned.

        If both :math:`y\_pred` and :math:`y\_true` are
        :class:`~skfda.representation.FDataGrid`
        objects and multioutput = 'raw_values',
        :class:`~skfda.representation.FDataGrid` is returned.

        If both :math:`y\_pred` and :math:`y\_true` are ndarray and
        multioutput = 'raw_values', ndarray.

    """
    return sklearn.metrics.r2_score(  # type: ignore [no-any-return]
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@r2_score.register  # type: ignore[attr-defined, misc]
def _r2_score_fdatagrid(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> Union[float, FDataGrid]:
    from ..exploratory.stats import mean

    if y_pred.n_samples < 2:
        raise ValueError(
            'R^2 score is not well-defined with less than two samples.',
        )

    ss_res = mean(
        np.power(y_true - y_pred, 2),
        weights=sample_weight,
    )

    ss_tot = _var(y_true, weights=sample_weight)

    # Divisions by zero allowed
    with np.errstate(divide='ignore', invalid='ignore'):
        score: FDataGrid = 1 - ss_res / ss_tot

    # 0 / 0 divisions should be 0 in this context and the score, 1
    score.data_matrix[np.isnan(score.data_matrix)] = 1

    return _multioutput_score_grid(score, multioutput)


@r2_score.register  # type: ignore[attr-defined, misc]
def _r2_score_fdatabasis(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: MultiOutputType = 'uniform_average',
) -> float:

    if y_pred.n_samples < 2:
        raise ValueError(
            'R^2 score is not well-defined with less than two samples.',
        )

    def _r2_func(x: NDArrayFloat) -> NDArrayFloat:  # noqa: WPS430
        ss_res = np.average(
            (y_true(x) - y_pred(x)) ** 2,
            weights=sample_weight,
            axis=0,
        )

        ss_tot = np.average(
            (
                y_true(x)
                - np.average(y_true(x), weights=sample_weight, axis=0)
            ) ** 2,
            weights=sample_weight,
            axis=0,
        )

        # Divisions by zero allowed
        with np.errstate(divide='ignore', invalid='ignore'):
            score = 1 - ss_res / ss_tot

        # 0/0 case, the score is 1.
        score[np.isnan(score)] = 1

        # r/0 case, r!= 0. Return -inf outside this function
        if np.any(np.isinf(score)):
            raise _InfiniteScoreError

        # Score only had 1 input point
        assert score.shape[0] == 1
        return score[0]  # type: ignore [no-any-return]

    return _multioutput_score_basis(y_true, multioutput, _r2_func)
