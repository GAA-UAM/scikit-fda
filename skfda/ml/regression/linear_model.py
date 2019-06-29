from skfda.representation.basis import *
from skfda.misc._lfd import LinearDifferentialOperator as Lfd

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted

from skfda.representation.basis import FDataBasis, Constant, Basis, FData

import numpy as np

class LinearFunctionalRegression(BaseEstimator, RegressorMixin):

    def __init__(self, beta_basis):
        self.beta_basis = beta_basis

    def fit(self, X, y=None, sample_weight=None):

        y, X, weights = self._argcheck(y, X, sample_weight)

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

    def _argcheck(self, y, x, weights = None):
        """Do some checks to types and shapes"""
        if not isinstance(y, FData):
            raise ValueError("The explanined variable is not an FData objetc")

        # TODO check for same domain_range

        xlen = len(x)
        blen = len(self.beta_basis)

        if blen != xlen:
            raise ValueError("Independent variables number should be equal "
                             "to the number of beta basis")

        for j in range(0, xlen):
            if isinstance(x[j], list):
                xjcoefs = np.asarray(x[j]).reshape((-1, 1))
                x[j] = FDataBasis(Constant(y.domain_range), xjcoefs)

        if any(y.nsamples != xfd.nsamples for xfd in x):
            raise ValueError("Dependent and independent variables should "
                             "have the same number of samples")

        if any(not isinstance(b, Basis) for b in self.beta_basis):
            raise ValueError("Betas should be a list of Basis.")

        if weights is None:
            weights = [1 for _ in range(y.nsamples)]

        if len(weights) != y.nsamples:
            raise ValueError("The number of weights should be equal to the "
                             "independent samples.")

        if np.any(np.array(weights) < 0):
            raise ValueError("The weights should be non negative values")

        return y, x, weights
