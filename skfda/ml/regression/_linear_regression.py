from __future__ import annotations

import itertools
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ..._utils import function_to_fdatabasis, nquad_vec
from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.lstsq import solve_regularized_weighted_lstsq
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData, FDataBasis
from ...representation.basis import Basis, ConstantBasis
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
    RegressorMixin[
        Union[AcceptedDataType, Sequence[AcceptedDataType]],
        NDArrayFloat,
    ],
    BaseEstimator,
):
    r"""Linear regression with multivariate and functional response.

    This is a regression algorithm equivalent to multivariate linear
    regression, but accepting also functional data expressed in a basis
    expansion.

    Functional linear regression model is subdivided into three broad
    categories, depending on whether the responses or the covariates,
    or both, are curves.

    Particulary, when the response is scalar, the model assumed is:

    .. math::
        y = w_0 + w_1 x_1 + \ldots + w_p x_p + \int w_{p+1}(t) x_{p+1}(t) dt \
        + \ldots + \int w_r(t) x_r(t) dt

    where the covariates can be either multivariate or functional and the
    response is multivariate.

    When the response is functional, the model assumed is:

    .. math::
        y(t) = \boldsymbol{X} \boldsymbol{\beta}(t)

    where the covariates are multivariate and the response is functional; or:

    .. math::
        y(t) = \boldsymbol{X}(t) \boldsymbol{\beta}(t)

    if the model covariates are also functional (concurrent).


    .. deprecated:: 0.8.
        Usage of arguments of type sequence of FData, ndarray is deprecated
        in methods fit, predict.
        Use covariate parameters of type pandas.DataFrame instead.

    Warning:
        For now, only multivariate and concurrent convariates are supported
        when the response is functional.

    Warning:
        This functionality is still experimental. Users should be aware that
        the API will suffer heavy changes in the future.

    Args:
        coef_basis (iterable): Basis of the coefficient functions of the
            functional covariates.
            When the response is scalar, if multivariate data is supplied,
            their corresponding entries should be ``None``. If ``None`` is
            provided for a functional covariate, the same basis is assumed.
            If this parameter is ``None`` (the default), it is assumed that
            ``None`` is provided for all covariates.
            When the response is functional, if ``None`` is provided, the
            response basis is used for all covariates.
        fit_intercept:  Whether to calculate the intercept for this
            model. If set to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
            When the response is functional, a coef_basis for intercept must be
            supplied.
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
            covariate.
        intercept\_: Independent term in the linear model. Set to 0.0
            if `fit_intercept = False`.

    Examples:

        Functional linear regression can be used with functions expressed in
        a basis. Also, a functional basis for the weights can be specified:

        >>> from skfda.ml.regression import LinearRegression
        >>> from skfda.representation.basis import (
        ...     FDataBasis,
        ...     MonomialBasis,
        ...     ConstantBasis,
        ... )

        >>> x_basis = MonomialBasis(n_basis=3)
        >>> X_train = FDataBasis(
        ...     basis=x_basis,
        ...     coefficients=[
        ...         [0, 0, 1],
        ...         [0, 1, 0],
        ...         [0, 1, 1],
        ...         [1, 0, 1],
        ...     ],
        ... )
        >>> y_train = [2, 3, 4, 5]
        >>> X_test = X_train  # Just for illustration purposes
        >>> linear_reg = LinearRegression()
        >>> _ = linear_reg.fit(X_train, y_train)
        >>> linear_reg.coef_[0]
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
            coefficients=[[-15.  96. -90.]],
            ...)
        >>> linear_reg.intercept_
        array([ 1.])
        >>> linear_reg.predict(X_test)
        array([ 2.,  3.,  4.,  5.])

        Covariates can include also multivariate data.
        In order to mix functional and multivariate data a dataframe can
        be used:

        >>> import pandas as pd
        >>> x_basis = MonomialBasis(n_basis=2)
        >>> X_train = pd.DataFrame({
        ...     "functional_covariate": FDataBasis(
        ...         basis=x_basis,
        ...         coefficients=[
        ...             [0, 2],
        ...             [0, 4],
        ...             [1, 0],
        ...             [2, 0],
        ...             [1, 2],
        ...             [2, 2],
        ...         ]
        ...     ),
        ...     "multivariate_covariate_1": [1, 2, 4, 1, 3, 2],
        ...     "multivariate_covariate_2": [7, 3, 2, 1, 1, 5],
        ... })
        >>> y_train = [11, 10, 12, 6, 10, 13]
        >>> X_test = X_train  # Just for illustration purposes
        >>> linear_reg = LinearRegression(
        ...     coef_basis=[ConstantBasis(), None, None],
        ... )
        >>> _ = linear_reg.fit(X_train, y_train)
        >>> linear_reg.coef_[0]
        FDataBasis(
            basis=ConstantBasis(domain_range=((0.0, 1.0),), n_basis=1),
            coefficients=[[ 1.]],
            ...)
        >>> linear_reg.coef_[1]
        array([ 2.])
        >>> linear_reg.coef_[2]
        array([ 1.])
        >>> linear_reg.intercept_
        array([ 1.])
        >>> linear_reg.predict(X_test)
        array([ 11.,  10.,  12.,   6.,  10.,  13.])

        Response can be functional when covariates are multivariate:

        >>> y_basis = MonomialBasis(n_basis=3)
        >>> X_train = pd.DataFrame({
        ...     "covariate_1": [3, 5, 3],
        ...     "covariate_2": [4, 1, 2],
        ...     "covariate_3": [1, 6, 8],
        ... })
        >>> y_train = FDataBasis(
        ...     basis=y_basis,
        ...     coefficients=[
        ...         [47, 22, 24],
        ...         [43, 47, 39],
        ...         [40, 53, 51],
        ...     ],
        ... )
        >>> X_test = X_train  # Just for illustration purposes
        >>> linear_reg = LinearRegression(
        ...     regularization=None,
        ...     fit_intercept=False,
        ... )
        >>> _ = linear_reg.fit(X_train, y_train)
        >>> linear_reg.coef_[0]
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
            coefficients=[[ 6.  3.  1.]],
            ...)
        >>> linear_reg.predict(X_test)
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
            coefficients=[[ 47. 22. 24.]
                [ 43. 47. 39.]
                [ 40. 53. 51.]],
             ...)

        Concurrent function-on-function regression is also supported:

        >>> x_basis = MonomialBasis(n_basis=3)
        >>> y_basis = MonomialBasis(n_basis=2)
        >>> X_train = FDataBasis(
        ...     basis=x_basis,
        ...     coefficients=[
        ...         [0, 1, 0],
        ...         [2, 1, 0],
        ...         [5, 0, 1],
        ...     ],
        ... )
        >>> y_train = FDataBasis(
        ...     basis=y_basis,
        ...     coefficients=[
        ...         [1, 1],
        ...         [2, 1],
        ...         [3, 1],
        ...     ],
        ... )
        >>> X_test = X_train  # Just for illustration purposes
        >>> linear_reg = LinearRegression(
        ...     coef_basis=[y_basis],
        ...     fit_intercept=False,
        ... )
        >>> _ = linear_reg.fit(X_train, y_train)
        >>> linear_reg.coef_[0]
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=2),
            coefficients=[[ 0.68250377  0.09960705]],
            ...)
        >>> linear_reg.predict(X_test)
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=2),
            coefficients=[[-0.01660117  0.78211082]
                [ 1.34840637  0.98132492]
                [ 3.27884682  1.27018536]],
             ...)

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
        y: AcceptedDataType,
        sample_weight: Optional[NDArrayFloat] = None,
    ) -> LinearRegression:

        X_new, y, sample_weight, coef_info = self._argcheck_X_y(
            X,
            y,
            sample_weight,
            self.coef_basis,
        )

        regularization: RegularizationIterableType = self.regularization

        coef_basis = self._coef_basis

        if self.fit_intercept:
            new_x = np.ones((len(y), 1))
            X_new = [new_x] + list(X_new)

            intercept_basis = self._choose_beta_basis(
                coef_basis=None,
                x_basis=None,
                y_basis=y.basis if isinstance(y, FDataBasis) else None,
            )
            coef_basis = [intercept_basis] + coef_basis

            new_coef_info_list: List[AcceptedDataCoefsType] = [
                coefficient_info_from_covariate(
                    new_x,
                    y,
                    basis=intercept_basis,
                ),
            ]
            coef_info = new_coef_info_list + list(coef_info)

            if not self._functional_response:
                if isinstance(regularization, Iterable):
                    regularization = itertools.chain([None], regularization)
                elif regularization is not None:
                    regularization = (None, regularization)

        penalty_matrix = compute_penalty_matrix(
            basis_iterable=(c.basis for c in coef_info),
            regularization_parameter=1,
            regularization=regularization,
        )

        if self._functional_response:
            coef_lengths = []
            left_inner_products_list = []
            right_inner_products_list = []

            Xt = self._make_transpose(X_new)

            for i, basis_i in enumerate(coef_basis):
                coef_lengths.append(basis_i.n_basis)
                row = []
                right_inner_products_list.append(
                    self._weighted_inner_product_integrate(
                        basis_i,
                        ConstantBasis(),
                        Xt[i],
                        y,
                    ),
                )
                for j, basis_j in enumerate(coef_basis):
                    row.append(
                        self._weighted_inner_product_integrate(
                            basis_i,
                            basis_j,
                            Xt[i],
                            Xt[j],
                        ),
                    )
                left_inner_products_list.append(row)

            coef_lengths.pop()
            left_inner_products = np.block(left_inner_products_list)
            right_inner_products = np.concatenate(right_inner_products_list)

            if penalty_matrix is not None:
                left_inner_products += penalty_matrix

            basiscoefs = np.linalg.solve(
                left_inner_products,
                right_inner_products,
            )
        else:
            if self.fit_intercept and penalty_matrix is not None:
                # Intercept is not penalized
                penalty_matrix[0, 0] = 0

            inner_products_list = [
                c.regression_matrix(x, y)  # type: ignore[arg-type]
                for x, c in zip(X_new, coef_info)
            ]

            # This is C @ J
            inner_products = np.concatenate(inner_products_list, axis=1)

            if sample_weight is not None:
                inner_products = inner_products * np.sqrt(sample_weight)
                y = y * np.sqrt(sample_weight)

            basiscoefs = solve_regularized_weighted_lstsq(
                coefs=inner_products,
                result=y,
                penalty_matrix=penalty_matrix,
            )

            coef_lengths = np.array([k.shape[1] for k in inner_products_list])

        coef_start = np.cumsum(coef_lengths)
        basiscoef_list = np.split(basiscoefs, coef_start)

        # Express the coefficients in functional form
        coefs = self._convert_from_constant_coefs(
            basiscoef_list,
            coef_info,
            coef_basis,
        )

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            coefs = coefs[1:]
        else:
            self.intercept_ = np.zeros(1)

        self.coef_ = coefs
        self.basis_coefs = basiscoef_list
        self._coef_info = coef_info
        self._target_ndim = y.ndim

        return self

    def predict(  # noqa: D102
        self,
        X: Union[Sequence[AcceptedDataType], pd.DataFrame],
    ) -> NDArrayFloat:

        check_is_fitted(self)
        X = self._argcheck_X(X)
        result = []

        for coef, x, coef_info in zip(self.coef_, X, self._coef_info):
            if self._functional_response:
                def prediction(arg, x_eval=x, coef_eval=coef):  # noqa: WPS430
                    if isinstance(x_eval, Callable):
                        x_eval = x_eval(arg)  # noqa: WPS220

                    # TODO: MIRAR ESTO BIEN
                    x_eval = x_eval.reshape((-1, 1, 1))

                    return coef_eval(arg) * x_eval

                result.append(
                    function_to_fdatabasis(prediction, self._y_basis),
                )
            else:
                result.append(coef_info.inner_product(coef, x))

        result = sum(result[1:], start=result[0])

        if self.fit_intercept:
            if self._functional_response:
                result = result + function_to_fdatabasis(
                    self.intercept_, self._y_basis,
                )
            else:
                result += self.intercept_

        if self._target_ndim == 1:
            result = result.ravel()

        return result

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

        return ([
            x if isinstance(x, FData)
            else self._check_and_convert(x) for x in X
        ])

    def _weighted_inner_product_integrate(
        self,
        basis: Union[FDataBasis, Basis],
        transposed_basis: Union[FDataBasis, Basis],
        transposed_weight: Union[FDataBasis, NDArrayFloat],
        weight: Union[FDataBasis, NDArrayFloat],
    ) -> NDArrayFloat:
        r"""
        Return the weighted inner product matrix between its arguments.

        For two basis (:math:\theta_1) and (:math:\theta_2) the weighted
        inner product is defined as:

        . math::
            \int \boldsymbol{\theta}_1 (t) \boldsymbol{\theta}^T_2 (t) w(t) dt

        where w(t) is defined by a functional vectors product:

        . math::
            \boldsymbol{w}^T_1 (t) \boldsymbol{w}_2 (t)

        Args:
            basis: Vertical basis vector.
            transposed_basis: Horizontal basis vector.
            transposed_weight: First weight, horizontal functional vector.
            weight: Second weight, vertical functional vector.

        Returns:
            Inner product matrix between basis and weight.

        """
        if isinstance(basis, Basis):
            basis = basis.to_basis()

        if isinstance(transposed_basis, Basis):
            transposed_basis = transposed_basis.to_basis()

        if isinstance(transposed_weight, FData) and not np.array_equal(
            basis.domain_range,
            transposed_weight.domain_range,
        ):
            raise ValueError("Domain range for weight and basis must be equal")

        domain_range = basis.domain_range

        def integrand(args):  # noqa: WPS430
            eval_basis = basis(args)[:, 0, :]
            eval_transposed_basis = transposed_basis(args)[:, 0, :]

            if isinstance(transposed_weight, FData):
                eval_transposed_weight = transposed_weight(args)[:, 0, :]
            else:
                eval_transposed_weight = transposed_weight.T

            if isinstance(weight, FData):
                eval_weight = weight(args)[:, 0, :]
            else:
                eval_weight = weight.T

            return eval_basis * eval_transposed_basis.T * np.dot(
                eval_transposed_weight.T, eval_weight,
            )

        return nquad_vec(
            integrand,
            domain_range,
        )

    def _make_transpose(
        self,
        X: Sequence[AcceptedDataType],
    ) -> Sequence[AcceptedDataType]:
        Xt: list[AcceptedDataType] = []
        for x in X:
            if isinstance(x, FData):
                Xt.append(x)
            else:
                x_new = self._check_and_convert(x)
                Xt.append(x_new.T)

        return Xt

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
        if new_X.ndim == 1:
            new_X = new_X[:, np.newaxis]
        return new_X

    def _convert_from_constant_coefs(
        self,
        coef_list: Sequence[NDArrayFloat],
        coef_info: Sequence[AcceptedDataCoefsType],
        coef_basis: Sequence[Basis],
    ) -> Sequence[FDataBasis]:
        if self._functional_response:
            return [
                FDataBasis(basis=basis, coefficients=coef.T)
                for basis, coef in zip(coef_basis, coef_list)
            ]
        return [
            c.convert_from_constant_coefs(bcoefs)
            for c, bcoefs in zip(coef_info, coef_list)
        ]

    def _argcheck_X_y(  # noqa: N802
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType], pd.DataFrame],
        y: Union[AcceptedDataType, Sequence[AcceptedDataType]],
        sample_weight: Optional[NDArrayFloat] = None,
        coef_basis: Optional[BasisCoefsType] = None,
    ) -> ArgcheckResultType:
        """Do some checks to types and shapes."""
        new_X = self._argcheck_X(X)

        len_new_X = len(new_X)

        self._y_basis = y.basis if isinstance(y, FDataBasis) else None

        if isinstance(y, FDataBasis):
            self._functional_response = True
        else:
            if any(len(y) != len(x) for x in new_X):
                raise ValueError(
                    "The number of samples on independent and "
                    "dependent variables should be the same",
                )
            self._functional_response = False
            y = np.asarray(y)

        if coef_basis is None:
            coef_basis = [None] * len_new_X

        if len(coef_basis) == 1 and len_new_X > 1:
            # we assume basis objects are inmmutable
            coef_basis = [coef_basis[0]] * len_new_X

        if len_new_X != len(coef_basis):
            raise ValueError(
                "The number of covariates and coefficients "
                "basis should be the same",
            )

        coef_basis = [
            self._choose_beta_basis(
                coef_basis=c_basis,
                x_basis=x.basis if isinstance(x, FDataBasis) else None,
                y_basis=self._y_basis,
            )
            for c_basis, x in zip(coef_basis, new_X)
        ]

        self._coef_basis = coef_basis

        coef_info = [
            coefficient_info_from_covariate(x, y, basis=b)
            for x, b in zip(new_X, coef_basis)
        ]

        if sample_weight is not None:
            self._sample_weight_check(sample_weight, y)

        return new_X, y, sample_weight, coef_info

    def _choose_beta_basis(
        self,
        coef_basis: Basis | None = None,
        x_basis: Basis | None = None,
        y_basis: Basis | None = None,
    ) -> Basis | None:
        if coef_basis is not None:
            return coef_basis
        elif y_basis is None:
            return x_basis
        else:
            return y_basis

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
