from __future__ import annotations

import itertools
import warnings
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.lstsq import solve_regularized_weighted_lstsq
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis
from ...typing._numpy import NDArrayFloat
from ._coefficients import CoefficientInfo, coefficient_info_from_covariate

RegularizationType = Union[
    L2Regularization[Any],
    Sequence[Optional[L2Regularization[Any]]],
    None,
]

RegularizationIterableType = Union[
    L2Regularization[Any],
    Iterable[Optional[L2Regularization[Any]]],
    None,
]

AcceptedDataType = Union[
    FData,
    NDArrayFloat,
]

AcceptedDataCoefsType = Union[
    CoefficientInfo[FData],
    CoefficientInfo[NDArrayFloat],
]

BasisCoefsType = Sequence[Optional[Basis]]


ArgcheckResultType = Tuple[
    Sequence[AcceptedDataType],
    NDArrayFloat,
    Optional[NDArrayFloat],
    Sequence[AcceptedDataCoefsType],
]


class LinearRegression(
    BaseEstimator,
    RegressorMixin[
        Union[AcceptedDataType, Sequence[AcceptedDataType]],
        NDArrayFloat,
    ],
):
    r"""Linear regression with multivariate response.

    This is a regression algorithm equivalent to multivariate linear
    regression, but accepting also functional data expressed in a basis
    expansion.

    The model assumed by this method is:

    .. math::
        y = w_0 + w_1 x_1 + \ldots + w_p x_p + \int w_{p+1}(t) x_{p+1}(t) dt \
        + \ldots + \int w_r(t) x_r(t) dt

    where the covariates can be either multivariate or functional and the
    response is multivariate.

    .. deprecated:: 0.8.
        Usage of arguments of type sequence of FData, ndarray is deprecated
        in methods fit, predict.
        Use covariate parameters of type pandas.DataFrame instead.

    .. warning::
        For now, only scalar responses are supported.

    Args:
        coef_basis (iterable): Basis of the coefficient functions of the
            functional covariates. If multivariate data is supplied, their
            corresponding entries should be ``None``. If ``None`` is provided
            for a functional covariate, the same basis is assumed. If this
            parameter is ``None`` (the default), it is assumed that ``None``
            is provided for all covariates.
        fit_intercept:  Whether to calculate the intercept for this
            model. If set to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
        regularization (int, iterable or :class:`Regularization`): If it is
            not a :class:`Regularization` object, linear differential
            operator regularization is assumed. If it
            is an integer, it indicates the order of the
            derivative used in the computing of the penalty matrix. For
            instance 2 means that the differential operator is
            :math:`f''(x)`. If it is an iterable, it consists on coefficients
            representing the differential operator used in the computing of
            the penalty matrix. For instance the tuple (1, 0,
            numpy.sin) means :math:`1 + sin(x)D^{2}`. If not supplied this
            defaults to 2. Only used if penalty_matrix is
            ``None``.

    Attributes:
        coef\_: A list containing the weight coefficient for each
            covariate. For multivariate data, the covariate is a Numpy array.
            For functional data, the covariate is a FDataBasis object.
        intercept\_: Independent term in the linear model. Set to 0.0
            if `fit_intercept = False`.

    Examples:
        >>> from skfda.ml.regression import LinearRegression
        >>> from skfda.representation.basis import (FDataBasis, MonomialBasis,
        ...                                         ConstantBasis)
        >>> import pandas as pd

        Multivariate linear regression can be used with functions expressed in
        a basis. Also, a functional basis for the weights can be specified:

        >>> x_basis = MonomialBasis(n_basis=3)
        >>> x_fd = FDataBasis(x_basis, [[0, 0, 1],
        ...                             [0, 1, 0],
        ...                             [0, 1, 1],
        ...                             [1, 0, 1]])
        >>> y = [2, 3, 4, 5]
        >>> linear = LinearRegression()
        >>> _ = linear.fit(x_fd, y)
        >>> linear.coef_[0]
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
            coefficients=[[-15.  96. -90.]],
            ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict(x_fd)
        array([ 2.,  3.,  4.,  5.])

        Covariates can include also multivariate data:

        >>> x_basis = MonomialBasis(n_basis=2)
        >>> x_fd = FDataBasis(x_basis, [[0, 2],
        ...                             [0, 4],
        ...                             [1, 0],
        ...                             [2, 0],
        ...                             [1, 2],
        ...                             [2, 2]])
        >>> x = [[1, 7], [2, 3], [4, 2], [1, 1], [3, 1], [2, 5]]
        >>> y = [11, 10, 12, 6, 10, 13]
        >>> linear = LinearRegression(
        ...              coef_basis=[None, ConstantBasis()])
        >>> _ = linear.fit([x, x_fd], y)
        >>> linear.coef_[0]
        array([ 2.,  1.])
        >>> linear.coef_[1]
        FDataBasis(
        basis=ConstantBasis(domain_range=((0.0, 1.0),), n_basis=1),
        coefficients=[[ 1.]],
        ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict([x, x_fd])
        array([ 11.,  10.,  12.,   6.,  10.,  13.])

        Funcionality with pandas Dataframe.

        First example:

        >>> x_basis = MonomialBasis(n_basis=3)
        >>> x_fd = FDataBasis(x_basis, [[0, 0, 1],
        ...                             [0, 1, 0],
        ...                             [0, 1, 1],
        ...                             [1, 0, 1]])
        >>> cov_dict = { "fd": x_fd }
        >>> y = [2, 3, 4, 5]
        >>> df = pd.DataFrame(cov_dict)
        >>> linear = LinearRegression()
        >>> _ = linear.fit(df, y)
        >>> linear.coef_[0]
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
            coefficients=[[-15.  96. -90.]],
            ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict(df)
        array([ 2.,  3.,  4.,  5.])

        Second example:

        >>> x_basis = MonomialBasis(n_basis=2)
        >>> x_fd = FDataBasis(x_basis, [[0, 2],
        ...                             [0, 4],
        ...                             [1, 0],
        ...                             [2, 0],
        ...                             [1, 2],
        ...                             [2, 2]])
        >>> mult1 = np.asarray([1, 2, 4, 1, 3, 2])
        >>> mult2 = np.asarray([7, 3, 2, 1, 1, 5])
        >>> cov_dict = {"m1": mult1, "m2": mult2, "fd": x_fd}
        >>> df = pd.DataFrame(cov_dict)
        >>> y = [11, 10, 12, 6, 10, 13]
        >>> linear = LinearRegression(
        ...              coef_basis=[None, ConstantBasis(), ConstantBasis()])
        >>> _ = linear.fit(df, y)
        >>> linear.coef_[0]
        array([ 2.])
        >>> linear.coef_[1]
        array([ 1.])
        >>> linear.coef_[2]
        FDataBasis(
            basis=ConstantBasis(domain_range=((0.0, 1.0),), n_basis=1),
            coefficients=[[ 1.]],
            ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict(df)
        array([ 11.,  10.,  12.,   6.,  10.,  13.])

    """

    def __init__(
        self,
        *,
        coef_basis: Optional[BasisCoefsType] = None,
        fit_intercept: bool = True,
        regularization: RegularizationType = None,
    ) -> None:
        self.coef_basis = coef_basis
        self.fit_intercept = fit_intercept
        self.regularization = regularization

    def fit(  # noqa: D102
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType], pd.DataFrame],
        y: NDArrayFloat,
        sample_weight: Optional[NDArrayFloat] = None,
    ) -> LinearRegression:

        X_new, y, sample_weight, coef_info = self._argcheck_X_y(
            X,
            y,
            sample_weight,
            self.coef_basis,
        )

        regularization: RegularizationIterableType = self.regularization

        if self.fit_intercept:
            new_x = np.ones((len(y), 1))
            X_new = [new_x] + list(X_new)
            new_coef_info_list: List[AcceptedDataCoefsType] = [
                coefficient_info_from_covariate(new_x, y),
            ]
            coef_info = new_coef_info_list + list(coef_info)

            if isinstance(regularization, Iterable):
                regularization = itertools.chain([None], regularization)
            elif regularization is not None:
                regularization = (None, regularization)

        inner_products_list = [
            c.regression_matrix(x, y)  # type: ignore[arg-type]
            for x, c in zip(X_new, coef_info)
        ]

        # This is C @ J
        inner_products = np.concatenate(inner_products_list, axis=1)

        if sample_weight is not None:
            inner_products = inner_products * np.sqrt(sample_weight)
            y = y * np.sqrt(sample_weight)

        penalty_matrix = compute_penalty_matrix(
            basis_iterable=(c.basis for c in coef_info),
            regularization_parameter=1,
            regularization=regularization,
        )

        if self.fit_intercept and penalty_matrix is not None:
            # Intercept is not penalized
            penalty_matrix[0, 0] = 0

        basiscoefs = solve_regularized_weighted_lstsq(
            coefs=inner_products,
            result=y,
            penalty_matrix=penalty_matrix,
        )

        coef_lengths = np.array([i.shape[1] for i in inner_products_list])
        coef_start = np.cumsum(coef_lengths)
        basiscoef_list = np.split(basiscoefs, coef_start)

        # Express the coefficients in functional form
        coefs = [
            c.convert_from_constant_coefs(bcoefs)
            for c, bcoefs in zip(coef_info, basiscoef_list)
        ]

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            coefs = coefs[1:]
        else:
            self.intercept_ = np.zeros(1)

        self.coef_ = coefs
        self._coef_info = coef_info
        self._target_ndim = y.ndim

        return self

    def predict(  # noqa: D102
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType], pd.DataFrame],
    ) -> NDArrayFloat:

        check_is_fitted(self)
        X = self._argcheck_X(X)

        result = np.sum(
            [
                coef_info.inner_product(coef, x)  # type: ignore[arg-type]
                for coef, x, coef_info
                in zip(self.coef_, X, self._coef_info)
            ],
            axis=0,
        )

        result += self.intercept_

        if self._target_ndim == 1:
            result = result.ravel()

        return result  # type: ignore[no-any-return]

    def _argcheck_X(  # noqa: N802
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType], pd.DataFrame],
    ) -> Sequence[AcceptedDataType]:

        if isinstance(X, List) and any(isinstance(x, FData) for x in X):
            warnings.warn(
                "Usage of arguments of type sequence of "
                "FData, ndarray is deprecated (fit, predict). "
                "Use pandas DataFrame instead",
                DeprecationWarning,
            )

        if isinstance(X, (FData, np.ndarray)):
            X = [X]

        elif isinstance(X, pd.DataFrame):
            X = self._dataframe_conversion(X)

        X = [
            x if isinstance(x, FData)
            else self._check_and_convert(x) for x in X
        ]

        if all(not isinstance(i, FData) for i in X):
            warnings.warn("All the covariates are scalar.")

        return X

    def _check_and_convert(
        self,
        X: AcceptedDataType,
    ) -> np.ndarray:
        """Check if the input array is 1D and converts it to a 2D array.

        Args:
            X: multivariate array to check and convert.

        Returns:
            np.ndarray: numpy 2D array.
        """
        new_X = np.asarray(X)
        if len(new_X.shape) == 1:
            new_X = new_X[:, np.newaxis]
        return new_X

    def _argcheck_X_y(  # noqa: N802
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType], pd.DataFrame],
        y: NDArrayFloat,
        sample_weight: Optional[NDArrayFloat] = None,
        coef_basis: Optional[BasisCoefsType] = None,
    ) -> ArgcheckResultType:
        """Do some checks to types and shapes."""
        new_X = self._argcheck_X(X)

        if any(isinstance(i, FData) for i in y):
            raise ValueError(
                "Some of the response variables are not scalar",
            )

        y = np.asarray(y)

        if coef_basis is None:
            coef_basis = [None] * len(new_X)

        if len(coef_basis) != len(new_X):
            raise ValueError(
                "Number of regression coefficients does "
                "not match number of independent variables.",
            )

        if any(len(y) != len(x) for x in new_X):
            raise ValueError(
                "The number of samples on independent and "
                "dependent variables should be the same",
            )

        coef_info = [
            coefficient_info_from_covariate(x, y, basis=b)
            for x, b in zip(new_X, coef_basis)
        ]

        if sample_weight is not None:
            self._sample_weight_check(sample_weight, y)

        return new_X, y, sample_weight, coef_info

    def _sample_weight_check(
        self,
        sample_weight: Optional[NDArrayFloat],
        y: NDArrayFloat,
    ):
        if len(sample_weight) != len(y):
            raise ValueError(
                "The number of sample weights should be "
                "equal to the number of samples.",
            )

        if np.any(np.array(sample_weight) < 0):
            raise ValueError(
                "The sample weights should be non negative values",
            )

    def _dataframe_conversion(
        self,
        X: pd.DataFrame,
    ) -> List[AcceptedDataType]:
        """Convert DataFrames to a list with input columns.

        Args:
            X: pandas DataFrame to convert.

        Returns:
            List: list which elements are the input DataFrame columns.
        """
        return [v.values for k, v in X.items()]
