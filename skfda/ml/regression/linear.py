from skfda.misc._math import inner_product
from skfda.representation import FData
from skfda.representation.basis import FDataBasis, Constant, Basis

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np


class LinearScalarRegression(BaseEstimator, RegressorMixin):

    def __init__(self, *, coef_basis=None, fit_intercept=True):
        self.coef_basis = coef_basis
        self.fit_intercept = fit_intercept

    def fit(self, X, y=None, sample_weight=None):

        X, y, sample_weight, coef_basis = self._argcheck_X_y(
            X, y, sample_weight, self.coef_basis)

        if self.fit_intercept:
            X = [np.ones((len(y), 1))] + X
            coef_basis = [None] + coef_basis

        # X is a list of covariates
        n_covariates = len(X)

        inner_products = [None] * n_covariates

        for i, (x, w_basis) in enumerate(zip(X, coef_basis)):
            if isinstance(x, FDataBasis):
                if w_basis is None:
                    w_basis = x.basis
                xcoef = x.coefficients
                inner_basis = x.basis.inner_product(w_basis)
                inner = xcoef @ inner_basis
            else:
                if w_basis is not None:
                    raise ValueError("Multivariate data coefficients "
                                     "should not have a basis")
                inner = np.atleast_2d(x)
            inner_products[i] = inner

        # This is C @ J
        inner_products = np.concatenate(inner_products, axis=1)

        if any(w != 1 for w in sample_weight):
            inner_products = inner_products * np.sqrt(sample_weight)
            y = y * np.sqrt(sample_weight)

        gram_inner_x_coef = inner_products.T @ inner_products
        inner_x_coef_y = inner_products.T @ y

        coef_basiscoefs = np.linalg.solve(gram_inner_x_coef, inner_x_coef_y)

        # Express the coefficients in functional form
        coefs = [None] * n_covariates
        idx = 0
        for i, (x, basis) in enumerate(zip(X, coef_basis)):
            if isinstance(x, FDataBasis):
                if basis is None:
                    basis = x.basis

                # Functional coefs
                used_coefs = basis.n_basis
                coefs[i] = FDataBasis(
                    basis,
                    coef_basiscoefs[idx:idx + used_coefs].T)
            else:
                # Multivariate coefs
                used_coefs = x.shape[1]
                coefs[i] = coef_basiscoefs[idx:idx + used_coefs]
            idx = idx + used_coefs

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

        if all(not isinstance(i, FData) for i in X):
            raise ValueError("All the covariates are scalar.")

        return X

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
            raise ValueError("Number of regression coefficients does"
                             " not match number of independent variables.")

        if any(len(y) != len(x) for x in X):
            raise ValueError("The number of samples on independent and "
                             "dependent variables should be the same")

        if any(b is not None and not isinstance(b, Basis)
               for b in coef_basis):
            raise ValueError("coefs should be a list of Basis.")

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        if len(sample_weight) != len(y):
            raise ValueError("The number of sample weights should be equal to"
                             "the number of samples.")

        if np.any(np.array(sample_weight) < 0):
            raise ValueError(
                "The sample weights should be non negative values")

        return X, y, sample_weight, coef_basis
