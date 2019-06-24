from sklearn.metrics import mean_squared_error

from skfda.representation.basis import *
from skfda.misc._lfd import LinearDifferentialOperator as Lfd

from sklearn.base import BaseEstimator, RegressorMixin
from skfda.representation.basis import FDataBasis, Constant, Basis, FData

import numpy as np


class LinearScalarRegression(BaseEstimator, RegressorMixin):

    def __init__(self, beta, weights=None):
        self.beta_ = None
        self.beta = beta
        self.weights = weights

    def fit(self, X, y):

        y, X, beta, weights = self._argcheck(y, X)

        nbeta = len(beta)
        nsamples = X[0].nsamples

        y = np.array(y).reshape((nsamples, 1))

        for j in range(nbeta):
            xcoef = X[j].coefficients
            inner_basis_x_beta_j = X[j].basis.inner_product(beta[j])
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
            beta[j] = FDataBasis(beta[j], betacoefs[idx:idx+beta[j].nbasis].T)
            idx = idx + beta[j].nbasis

        self.beta_ = beta
        return self

    def predict(self, X):
        return [sum(self.beta[i].inner_product(X[i][j])[0, 0] for i in
                    range(len(self.beta))) for j in range(X[0].nsamples)]

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def _argcheck(self, y, x):
        """Do some checks to types and shapes"""
        if all(not isinstance(i, FData) for i in x):
            raise ValueError("All the dependent variable are scalar.")
        if any(isinstance(i, FData) for i in y):
            raise ValueError(
                "Some of the independent variables are not scalar")

        ylen = len(y)
        xlen = len(x)
        blen = len(self.beta)
        domain_range = ([i for i in x if isinstance(i, FData)][0]
                        .domain_range)

        if blen != xlen:
            raise ValueError("Number of regression coefficients does"
                             " not match number of independent variables.")

        for j in range(xlen):
            if isinstance(x[j], list):
                xjcoefs = np.array(x[j]).reshape((-1, 1))
                x[j] = FDataBasis(Constant(domain_range), xjcoefs)

        if any(ylen != xfd.nsamples for xfd in x):
            raise ValueError("The number of samples on independent and "
                             "dependent variables should be the same")

        if any(not isinstance(b, Basis) for b in self.beta):
            raise ValueError("Betas should be a list of Basis.")

        if self.weights is None:
            self.weights = [1 for _ in range(ylen)]

        if len(self.weights) != ylen:
            raise ValueError("The number of weights should be equal to the "
                             "independent samples.")

        if np.any(np.array(self.weights) < 0):
            raise ValueError("The weights should be non negative values")

        return y, x, self.beta, self.weights


class LinearFunctionalRegression(BaseEstimator, RegressorMixin):

    def __init__(self, beta, weights=None):
        self.beta = beta
        self.weights = weights

    def fit(self, X, y):

        y, X = self._argcheck(y, X)

        rangeval = y.domain_range[0]
        onesfd = FDataBasis(Constant(rangeval), [1])

        betaindex = np.concatenate((np.array([0]), np.add.accumulate([i.nbasis for i in self.beta])))
        betacoefindex = [list(range(betaindex[i-1], betaindex[i])) for i in range(1, len(betaindex))]

        ncoef = sum(self.beta[i].nbasis for i in range(len(self.beta)))

        Cmat = np.zeros((ncoef, ncoef))
        Dmat = np.zeros((ncoef, 1))

        mj2 = 0
        for j in range(len(X)):
            mj1 = mj2
            mj2 = mj2 + self.beta[j].nbasis
            xyfdj = X[j].times(self.weights).times(y)
            wtfdj = sum(xyfdj)
            Dmat[mj1:mj2] = inprod(self.beta[j].to_basis(), onesfd,
                                   Lfd(0), Lfd(0), rangeval, wtfdj)

            mk2 = 0
            for k in range(0, j + 1):
                mk1 = mk2
                mk2 = mk2 + self.beta[k].nbasis
                xxfdjk = X[j].times(self.weights).times(X[k])
                wtfdjk = sum(xxfdjk)
                Cmatjk = inprod(self.beta[j].to_basis(), self.beta[k].to_basis(),
                                Lfd(0), Lfd(0), rangeval, wtfdjk)

                Cmat[mj1: mj2, mk1: mk2] = Cmatjk
                Cmat[mk1: mk2, mj1: mj2] = np.transpose(Cmatjk)

        Cmat = (Cmat + np.transpose(Cmat)) / 2
        Cmatinv = np.linalg.inv(Cmat)
        betacoef = np.transpose(np.transpose(Cmatinv) @ Dmat)[0]

        mj2 = 0
        for j in range(len(self.beta)):
            mj1 = mj2
            mj2 = mj2 + self.beta[j].nbasis
            coefj = betacoef[mj1: mj2]
            self.beta[j] = FDataBasis(self.beta[j].copy(), coefj)

    # TODO rehacer este predict
    def predict(self, X):
        return [sum(self.beta[i].inner_product(X[i][j])[0, 0] for i in
                    range(len(self.beta))) for j in range(X[0].nsamples)]


    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)

    def _argcheck(self, y, x):
        if all(not isinstance(i, FDataBasis) for i in x):
            domain_range = y.domain_range
        else:
            domain_range = ([i for i in x if isinstance(i, FDataBasis)]
                            [0].domain_range)

        ylen = y.nsamples
        xlen = len(x)
        blen = len(self.beta)

        if blen != xlen:
            raise ValueError("Independent variables number should be equal "
                             "to the number of beta basis")

        for j in range(0, xlen):
            if isinstance(x[j], list):
                xjcoefs = np.array(x[j]).reshape((-1, 1))
                x[j] = FDataBasis(Constant(domain_range), xjcoefs)

        if any(ylen != xfd.nsamples for xfd in x):
            raise ValueError("Dependent and independent variables should "
                             "have the same number of samples")

        if any(not isinstance(b, Basis) for b in self.beta):
            raise ValueError("Betas should be a list of Basis.")

        if self.weights is None:
            self.weights = [1 for _ in range(ylen)]

        if len(self.weights) != ylen:
            raise ValueError("The number of weights should be equal to the "
                             "independent samples.")

        if np.any(np.array(self.weights) < 0):
            raise ValueError("The weights should be non negative values")

        return y, x
