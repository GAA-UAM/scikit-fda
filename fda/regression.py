from sklearn.metrics import mean_squared_error

from fda.basis import *
import numpy as np
from fda import *


class FunctionalRegression:

    def __init__(self, beta, weights=None):
        self.beta = beta
        self.weights = weights

    def fit(self, y, x):

        y, x = self._argcheck(y, x)

        rangeval = y.domain_range[0]
        onesfd = FDataBasis(Constant(rangeval), [1])

        betaindex = np.concatenate((np.array([0]), np.add.accumulate([i.nbasis for i in self.beta])))
        betacoefindex = [list(range(betaindex[i-1], betaindex[i])) for i in range(1, len(betaindex))]

        ncoef = sum(self.beta[i].nbasis for i in range(len(self.beta)))

        Cmat = np.zeros((ncoef, ncoef))
        Dmat = np.zeros((ncoef, 1))

        mj2 = 0
        for j in range(len(x)):
            mj1 = mj2
            mj2 = mj2 + self.beta[j].nbasis
            xyfdj = x[j].times(self.weights).times(y)
            wtfdj = xyfdj.plus_samples()
            Dmat[mj1:mj2] = inprod(self.beta[j].to_basis(), onesfd,
                                   Lfd(0), Lfd(0), rangeval, wtfdj)

            mk2 = 0
            for k in range(0, j + 1):
                mk1 = mk2
                mk2 = mk2 + self.beta[k].nbasis
                xxfdjk = x[j].times(self.weights).times(x[k])
                wtfdjk = xxfdjk.plus_samples()
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
    def predict(self, x):
        return [sum(self.beta[i].inner_product(x[i][j])[0, 0] for i in
                    range(len(self.beta))) for j in range(x[0].nsamples)]

    def mean_squared_error(self, y_actual, y_predicted):
        return mean_squared_error(y_actual, y_predicted)

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
