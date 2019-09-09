from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
from skfda.representation.basis import FDataBasis, Constant, Basis, FData


class LinearScalarRegression(BaseEstimator, RegressorMixin):

    def __init__(self, beta_basis):
        self.beta_basis = beta_basis

    def fit(self, X, y=None, sample_weight=None):

        y, X, weights = self._argcheck(y, X, sample_weight)

        nbeta = len(self.beta_basis)
        n_samples = X[0].n_samples

        y = np.asarray(y).reshape((n_samples, 1))

        for j in range(nbeta):
            xcoef = X[j].coefficients
            inner_basis_x_beta_j = X[j].basis.inner_product(self.beta_basis[j])
            inner_x_beta = (xcoef @ inner_basis_x_beta_j
                            if j == 0
                            else np.concatenate((inner_x_beta,
                                                 xcoef @ inner_basis_x_beta_j),
                                                axis=1))

        if any(w != 1 for w in weights):
            inner_x_beta = inner_x_beta * np.sqrt(weights)
            y = y * np.sqrt(weights)

        gram_inner_x_beta = inner_x_beta.T @ inner_x_beta
        inner_x_beta_y = inner_x_beta.T @ y

        gram_inner_x_beta_inv = np.linalg.inv(gram_inner_x_beta)
        betacoefs = gram_inner_x_beta_inv @ inner_x_beta_y

        idx = 0
        for j in range(0, nbeta):
            self.beta_basis[j] = FDataBasis(
                self.beta_basis[j],
                betacoefs[idx:idx + self.beta_basis[j].n_basis].T)
            idx = idx + self.beta_basis[j].n_basis

        self.beta_ = self.beta_basis
        return self

    def predict(self, X):
        check_is_fitted(self, "beta_")
        return [sum(self.beta[i].inner_product(X[i][j])[0, 0] for i in
                    range(len(self.beta))) for j in range(X[0].n_samples)]

    def _argcheck(self, y, x, weights=None):
        """Do some checks to types and shapes"""
        if all(not isinstance(i, FData) for i in x):
            raise ValueError("All the dependent variable are scalar.")
        if any(isinstance(i, FData) for i in y):
            raise ValueError(
                "Some of the independent variables are not scalar")

        ylen = len(y)
        xlen = len(x)
        blen = len(self.beta_basis)
        domain_range = ([i for i in x if isinstance(i, FData)][0]
                        .domain_range)

        if blen != xlen:
            raise ValueError("Number of regression coefficients does"
                             " not match number of independent variables.")

        for j in range(xlen):
            if isinstance(x[j], list):
                xjcoefs = np.array(x[j]).reshape((-1, 1))
                x[j] = FDataBasis(Constant(domain_range), xjcoefs)

        if any(ylen != xfd.n_samples for xfd in x):
            raise ValueError("The number of samples on independent and "
                             "dependent variables should be the same")

        if any(not isinstance(b, Basis) for b in self.beta_basis):
            raise ValueError("Betas should be a list of Basis.")

        if weights is None:
            weights = [1 for _ in range(ylen)]

        if len(weights) != ylen:
            raise ValueError("The number of weights should be equal to the "
                             "independent samples.")

        if np.any(np.array(weights) < 0):
            raise ValueError("The weights should be non negative values")

        return y, x, weights
