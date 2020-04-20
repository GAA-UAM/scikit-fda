from skfda.misc._math import inner_product
from skfda.representation import FData
from skfda.representation.basis import FDataBasis, Constant, Basis

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np


class MultivariateLinearRegression(BaseEstimator, RegressorMixin):
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

    .. warning::
        For now, only scalar responses are supported.

    Args:
        coef_basis (iterable): Basis of the coefficient functions of the
            functional covariates. If multivariate data is supplied, their
            corresponding entries should be ``None``. If ``None`` is provided
            for a functional covariate, the same basis is assumed. If this
            parameter is ``None`` (the default), it is assumed that ``None``
            is provided for all covariates.
        fit_intercept (bool):  Whether to calculate the intercept for this
            model. If set to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
        regularization_parameter (int or float, optional): Regularization
            parameter. Trying with several factors in a logarithm scale is
            suggested. If 0 no regularization is performed. Defaults to 0.
        penalty (int, iterable or :class:`LinearDifferentialOperator`): If it
            is an integer, it indicates the order of the
            derivative used in the computing of the penalty matrix. For
            instance 2 means that the differential operator is
            :math:`f''(x)`. If it is an iterable, it consists on coefficients
            representing the differential operator used in the computing of
            the penalty matrix. For instance the tuple (1, 0,
            numpy.sin) means :math:`1 + sin(x)D^{2}`. It is possible to
            supply directly the LinearDifferentialOperator object.
            If not supplied this defaults to 2. Only used if penalty_matrix is
            ``None``.
        penalty_matrix (array_like, optional): Penalty matrix. If
            supplied the differential operator is not used and instead
            the matrix supplied by this argument is used.

    Attributes:
        coef_ (iterable): A list containing the weight coefficient for each
            covariate. For multivariate data, the covariate is a Numpy array.
            For functional data, the covariate is a FDataBasis object.
        intercept_ (float): Independent term in the linear model. Set to 0.0
            if `fit_intercept = False`.

    Examples:

        >>> from skfda.ml.regression import MultivariateLinearRegression
        >>> from skfda.representation.basis import FDataBasis, Monomial

        Multivariate linear regression can be used with functions expressed in
        a basis. Also, a functional basis for the weights can be specified:

        >>> x_basis = Monomial(n_basis=3)
        >>> x_fd = FDataBasis(x_basis, [[0, 0, 1],
        ...                             [0, 1, 0],
        ...                             [0, 1, 1],
        ...                             [1, 0, 1]])
        >>> y = [2, 3, 4, 5]
        >>> linear = MultivariateLinearRegression()
        >>> _ = linear.fit(x_fd, y)
        >>> linear.coef_[0]
        FDataBasis(
            basis=Monomial(domain_range=[array([0, 1])], n_basis=3),
            coefficients=[[-15.  96. -90.]],
            ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict(x_fd)
        array([ 2.,  3.,  4.,  5.])

        Covariates can include also multivariate data:

        >>> x_basis = Monomial(n_basis=2)
        >>> x_fd = FDataBasis(x_basis, [[0, 2],
        ...                             [0, 4],
        ...                             [1, 0],
        ...                             [2, 0],
        ...                             [1, 2],
        ...                             [2, 2]])
        >>> x = [[1, 7], [2, 3], [4, 2], [1, 1], [3, 1], [2, 5]]
        >>> y = [11, 10, 12, 6, 10, 13]
        >>> linear = MultivariateLinearRegression(
        ...              coef_basis=[None, Constant()])
        >>> _ = linear.fit([x, x_fd], y)
        >>> linear.coef_[0]
        array([ 2.,  1.])
        >>> linear.coef_[1]
        FDataBasis(
        basis=Constant(domain_range=[array([0, 1])], n_basis=1),
        coefficients=[[ 1.]],
        ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict([x, x_fd])
        array([ 11.,  10.,  12.,   6.,  10.,  13.])

    """

    def __init__(self, *, coef_basis=None, fit_intercept=True,
                 regularization_parameter=0,
                 penalty=None,
                 penalty_matrix=None):
        self.coef_basis = coef_basis
        self.fit_intercept = fit_intercept
        self.regularization_parameter = regularization_parameter
        self.penalty = penalty
        self.penalty_matrix = penalty_matrix

    def _inner_product_matrix(self, x, basis):
        """
        Compute the inner product matrix of a variable.

        The variable can be multivariate or functional.

        """
        if isinstance(x, FDataBasis):
            # Functional inner product
            xcoef = x.coefficients
            inner_basis = x.basis.inner_product(basis)
            return xcoef @ inner_basis
        else:
            # Multivariate inner product
            if basis is not None:
                raise ValueError("Multivariate data coefficients "
                                 "should not have a basis")
            return np.atleast_2d(x)

    def _convert_coefs(self, x, basis, coefs):
        """
        Convert to original form.
        """
        if isinstance(x, FDataBasis):
            # Functional coefs
            return FDataBasis(
                basis,
                coefs.T)
        else:
            # Multivariate coefs
            return coefs

    def fit(self, X, y=None, sample_weight=None):
        from ...misc.regularization import compute_penalty_matrix

        X, y, sample_weight, coef_basis = self._argcheck_X_y(
            X, y, sample_weight, self.coef_basis)

        if self.fit_intercept:
            X = [np.ones((len(y), 1))] + X
            coef_basis = [None] + coef_basis

        inner_products = [self._inner_product_matrix(x, basis)
                          for x, basis in zip(X, coef_basis)]

        coef_lengths = np.array([i.shape[1] for i in inner_products])
        coef_start = np.cumsum(coef_lengths)

        # This is C @ J
        inner_products = np.concatenate(inner_products, axis=1)

        if any(w != 1 for w in sample_weight):
            inner_products = inner_products * np.sqrt(sample_weight)
            y = y * np.sqrt(sample_weight)

        penalty_matrix = compute_penalty_matrix(
            X=X, basis=coef_basis,
            regularization_parameter=self.regularization_parameter,
            regularization=self.penalty,
            penalty_matrix=self.penalty_matrix)

        gram_inner_x_coef = inner_products.T @ inner_products + penalty_matrix
        inner_x_coef_y = inner_products.T @ y

        basiscoefs = np.linalg.solve(gram_inner_x_coef, inner_x_coef_y)
        basiscoef_list = np.split(basiscoefs, coef_start)

        # Express the coefficients in functional form
        coefs = [self._convert_coefs(x, basis, bcoefs)
                 for x, basis, bcoefs in zip(X, coef_basis, basiscoef_list)]

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            coefs = coefs[1:]
        else:
            self.intercept_ = 0.0

        self.coef_ = coefs
        self._target_ndim = y.ndim

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._argcheck_X(X)

        result = np.sum([self._inner_product_mixed(
            coef, x) for coef, x in zip(self.coef_, X)], axis=0)

        result += self.intercept_

        if self._target_ndim == 1:
            result = result.ravel()

        return result

    def _inner_product_mixed(self, x, y):
        inner_product = getattr(x, "inner_product", None)

        if inner_product is None:
            return y @ x
        else:
            return inner_product(y)

    def _argcheck_X(self, X):
        if isinstance(X, FData) or isinstance(X, np.ndarray):
            X = [X]

        X = [x if isinstance(x, FData) else np.asarray(x) for x in X]

        if all(not isinstance(i, FData) for i in X):
            raise ValueError("All the covariates are scalar.")

        return X

    def _get_coef_basis(self, x, basis):
        if basis is None:
            basis = getattr(x, 'basis', None)
            return basis
        else:
            if not isinstance(basis, Basis):
                raise ValueError("coef_basis should be a list of Basis.")
            return basis

    def _argcheck_X_y(self, X, y, sample_weight=None, coef_basis=None):
        """Do some checks to types and shapes"""

        # TODO: Add support for Dataframes

        X = self._argcheck_X(X)

        y = np.asarray(y)

        if (np.issubdtype(y.dtype, np.object_)
                and any(isinstance(i, FData) for i in y)):
            raise ValueError(
                "Some of the response variables are not scalar")

        if coef_basis is None:
            coef_basis = [None] * len(X)

        if len(coef_basis) != len(X):
            raise ValueError("Number of regression coefficients does "
                             "not match number of independent variables.")

        if any(len(y) != len(x) for x in X):
            raise ValueError("The number of samples on independent and "
                             "dependent variables should be the same")

        coef_basis = [self._get_coef_basis(x, b)
                      for x, b in zip(X, coef_basis)]

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        if len(sample_weight) != len(y):
            raise ValueError("The number of sample weights should be equal to"
                             "the number of samples.")

        if np.any(np.array(sample_weight) < 0):
            raise ValueError(
                "The sample weights should be non negative values")

        return X, y, sample_weight, coef_basis
