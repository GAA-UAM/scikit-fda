from skfda.representation import FData
from skfda.representation.basis import FDataBasis, Constant, Basis

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np


class LinearScalarRegression(BaseEstimator, RegressorMixin):

    def __init__(self, coef_basis):
        self.coef_basis = coef_basis

    def fit(self, X, y=None, sample_weight=None):

        X, y, sample_weight = self._argcheck_X_y(X, y, sample_weight)

        # X is a list of covariates
        n_covariates = len(X)

        inner_products = [None] * n_covariates

        for i, (x, w_basis) in enumerate(zip(X, self.coef_basis)):
            xcoef = x.coefficients
            inner_basis = x.basis.inner_product(w_basis)
            inner_products[i] = xcoef @ inner_basis

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
        for i, basis in enumerate(self.coef_basis):
            coefs[i] = FDataBasis(
                basis,
                coef_basiscoefs[idx:idx + basis.n_basis].T)
            idx = idx + basis.n_basis

        self.coef_ = coefs
        self._target_ndim = y.ndim

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._argcheck_X(X)

        inner_products = np.sum([covariate.inner_product(
            x) for covariate, x in zip(self.coef_, X)], axis=0)

        if self._target_ndim == 1:
            inner_products = inner_products.ravel()

        return inner_products

    def _argcheck_X(self, X):
        if isinstance(X, FData):
            X = [X]

        if all(not isinstance(i, FData) for i in X):
            raise ValueError("All the covariates are scalar.")

        domain_ranges = [x.domain_range for x in X if isinstance(x, FData)]
        domain_range = domain_ranges[0]

        for i, x in enumerate(X):
            if not isinstance(x, FData):
                # TODO: Support multivariate data
                coefs = np.asarray(x)
                X[i] = FDataBasis(Constant(domain_range), coefs)

        return X

    def _argcheck_X_y(self, X, y, sample_weight=None):
        """Do some checks to types and shapes"""

        # TODO: Add support for Dataframes

        X = self._argcheck_X(X)

        y = np.asarray(y)

        if any(isinstance(i, FData) for i in y):
            raise ValueError(
                "Some of the response variables are not scalar")

        if len(self.coef_basis) != len(X):
            raise ValueError("Number of regression coefficients does"
                             " not match number of independent variables.")

        if any(len(y) != len(x) for x in X):
            raise ValueError("The number of samples on independent and "
                             "dependent variables should be the same")

        if any(not isinstance(b, Basis) for b in self.coef_basis):
            raise ValueError("coefs should be a list of Basis.")

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        if len(sample_weight) != len(y):
            raise ValueError("The number of sample weights should be equal to"
                             "the number of samples.")

        if np.any(np.array(sample_weight) < 0):
            raise ValueError(
                "The sample weights should be non negative values")

        return X, y, sample_weight
