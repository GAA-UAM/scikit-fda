"""Score functions for FData."""
import warnings
from functools import singledispatch
from typing import Optional, Union

import numpy as np
import scipy.integrate
import sklearn
from typing_extensions import Literal

from .. import FData
from ..exploratory.stats import mean, var
from ..representation._typing import NDArrayFloat
from ..representation.basis import FDataBasis
from ..representation.grid import FDataGrid
from .metrics import l1_norm


def _domain_measure(fd: FData) -> float:
    measure = 1.0
    for interval in fd.domain_range:
        measure = measure * (interval[1] - interval[0])
    return measure


def _var(x, weight=None):
    if weight is None:
        return var(x)

    return mean(
        np.power(x - mean(x, weight=weight), 2),
        weight=weight,
    )


@singledispatch
def explained_variance_score(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid, NDArrayFloat]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return sklearn.metrics.explained_variance_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@explained_variance_score.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:

    num = _var(y_true - y_pred, weight=sample_weight)
    den = _var(y_true, weight=sample_weight)

    # Divisions by zero allowed
    with np.errstate(divide='ignore'):
        score = 1 - num / den

    # 0 / 0 divisions should be 1 in this context and the score, 0
    score.data_matrix[np.isnan(score.data_matrix)] = 0

    if multioutput == 'raw_values':
        return score

    return score.integrate()[0] / _domain_measure(score)


@explained_variance_score.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
) -> float:

    start, end = y_true.domain_range[0]

    def _ev_func(x):  # noqa: WPS430
        num = np.average(
            np.power(
                (
                    (y_true(x) - y_pred(x))
                    - np.average(
                        y_true(x) - y_pred(x),
                        weights=sample_weight,
                        axis=0,
                    )
                ),
                2,
            ),
            weights=sample_weight,
            axis=0,
        )

        den = np.average(
            np.power(
                (
                    y_true(x)
                    - np.average(y_true(x), weights=sample_weight, axis=0)
                ),
                2,
            ),
            weights=sample_weight,
            axis=0,
        )
        # Divisions by zero allowed
        with np.errstate(divide='ignore'):
            score = 1 - num / den

        # 0 / 0 divisions should be 1 in this context and the score, 0
        score[np.isnan(score)] = 0

        return score[0][0]

    with np.errstate(invalid='ignore'):
        integral = scipy.integrate.quad_vec(
            _ev_func,
            start,
            end,
        )

    return integral[0] / (end - start)


@singledispatch
def mean_absolute_error(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid, NDArrayFloat]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return sklearn.metrics.mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@mean_absolute_error.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:

    error = mean(np.abs(y_true - y_pred), weight=sample_weight)

    if multioutput == 'raw_values':
        return error

    return l1_norm(error) / _domain_measure(error)


@mean_absolute_error.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
) -> float:

    start, end = y_true.domain_range[0]

    def _mae_func(x):  # noqa: WPS430
        return np.average(
            np.abs(y_true(x) - y_pred(x)),
            weights=sample_weight,
            axis=0,
        )[0][0]

    integral = scipy.integrate.quad_vec(
        _mae_func,
        start,
        end,
    )

    return integral[0] / (end - start)


@singledispatch
def mean_absolute_percentage_error(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return sklearn.metrics.mean_absolute_percentage_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@mean_absolute_percentage_error.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:

    epsilon = np.finfo(np.float64).eps

    if np.any(np.abs(y_true.data_matrix) < epsilon):
        warnings.warn('Zero denominator', RuntimeWarning)

    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)

    error = mean(mape, weight=sample_weight)

    if multioutput == 'raw_values':
        return error

    return l1_norm(error) / _domain_measure(error)


@mean_absolute_percentage_error.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
) -> float:

    def _mape_func(x):  # noqa: WPS430

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
        return error[0][0]

    start, end = y_true.domain_range[0]
    integral = scipy.integrate.quad_vec(
        _mape_func,
        start,
        end,
    )

    return integral[0] / (end - start)


@singledispatch
def mean_squared_error(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
    squared: bool = True,
) -> Union[float, FDataGrid]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return mean_squared_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


@mean_squared_error.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
    squared: bool = True,
) -> Union[float, FDataGrid]:

    error = mean(
        np.power(y_true - y_pred, 2),
        weight=sample_weight,
    )

    if not squared:
        error = np.sqrt(error)

    if multioutput == 'raw_values':
        return error

    return l1_norm(error) / _domain_measure(error)


@mean_squared_error.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    squared: bool = True,
) -> float:

    start, end = y_true.domain_range[0]

    def _mse_func(x):  # noqa: WPS430

        error = np.average(
            np.power(y_true(x) - y_pred(x), 2),
            weights=sample_weight,
            axis=0,
        )

        if not squared:
            return np.sqrt(error)

        return error[0][0]

    integral = scipy.integrate.quad_vec(
        _mse_func,
        start,
        end,
    )

    return integral[0] / (end - start)


@singledispatch
def mean_squared_log_error(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
    squared: bool = True,
) -> Union[float, FDataGrid]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return sklearn.metrics.mean_squared_log_error(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


@mean_squared_log_error.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
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


@mean_squared_log_error.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    squared: bool = True,
) -> float:

    start, end = y_true.domain_range[0]

    def _msle_func(x):  # noqa: WPS430

        if np.any(y_true(x) < 0) or np.any(y_pred(x) < 0):
            raise ValueError(
                "Mean Squared Logarithmic Error cannot be used when "
                "targets functions have negative values.",
            )

        error = np.average(
            np.power(np.log1p(y_true(x)) - np.log1p(y_pred(x)), 2),
            weights=sample_weight,
            axis=0,
        )[0][0]

        if not squared:
            return np.sqrt(error)

        return error

    integral = scipy.integrate.quad_vec(
        _msle_func,
        start,
        end,
    )

    return integral[0] / (end - start)


@singledispatch
def r2_score(
    y_true: Union[FData, NDArrayFloat],
    y_pred: Union[FData, NDArrayFloat],
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:
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

        float:
            if multioutput = 'uniform_average' or
            :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataBasis` objects.
        FDataGrid:
            if both :math:`y\_pred` and :math:`y\_true` are
            :class:`~skfda.representation.FDataGrid`
            objects and multioutput = 'raw_values'.
        ndarray:
            if both :math:`y\_pred` and :math:`y\_true` are ndarray and
            multioutput = 'raw_values'.

    """
    return sklearn.metrics.r2_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@r2_score.register
def _(
    y_true: FDataGrid,
    y_pred: FDataGrid,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
    multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
) -> Union[float, FDataGrid]:

    if y_pred.n_samples < 2:
        raise ValueError(
            'R^2 score is not well-defined with less than two samples.',
        )

    ss_res = mean(
        np.power(y_true - y_pred, 2),
        weight=sample_weight,
    )

    ss_tot = mean(
        (y_true - mean(y_true, weight=sample_weight))
        * (y_true - mean(y_true, weight=sample_weight)),
        weight=sample_weight,
    )

    # Divisions by zero allowed
    with np.errstate(divide='ignore'):
        score = 1 - ss_res / ss_tot

    # 0 / 0 divisions should be 1 in this context and the score, 0
    score.data_matrix[np.isnan(score.data_matrix)] = 0

    if multioutput == 'raw_values':
        return score

    return score.integrate()[0] / _domain_measure(score)


@r2_score.register
def _(
    y_true: FDataBasis,
    y_pred: FDataBasis,
    *,
    sample_weight: Optional[NDArrayFloat] = None,
) -> float:

    start, end = y_true.domain_range[0]

    if y_pred.n_samples < 2:
        raise ValueError(
            'R^2 score is not well-defined with less than two samples.',
        )

    def _r2_func(x):  # noqa: WPS430
        ss_res = np.average(
            np.power(y_true(x) - y_pred(x), 2),
            weights=sample_weight,
            axis=0,
        )

        ss_tot = np.average(
            np.power(
                (
                    y_true(x)
                    - np.average(y_true(x), weights=sample_weight, axis=0)
                ),
                2,
            ),
            weights=sample_weight,
            axis=0,
        )
        # Divisions by zero allowed
        with np.errstate(divide='ignore'):
            score = 1 - ss_res / ss_tot

        # 0 / 0 divisions should be 1 in this context and the score, 0
        score[np.isnan(score)] = 0

        return score[0][0]

    with np.errstate(invalid='ignore'):
        integral = scipy.integrate.quad_vec(
            _r2_func,
            start,
            end,
        )
    return integral[0] / (end - start)
