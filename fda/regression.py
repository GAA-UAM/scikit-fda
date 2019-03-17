from sklearn.metrics import mean_squared_error

from fda.basis import *


class ScalarRegression:
    def __init__(self, beta, wt=None):
        self.beta = beta
        self.weights = wt

    def fit(self, y, x):

        y, x, beta, wt = self._argcheck(y, x)

        nbeta = len(beta)
        nsamples = x[0].nsamples

        y = np.array(y).reshape((nsamples, 1))

        Zmat = None
        Rmat = None

        for j in range(0, nbeta):
            xfdj = x[j]
            xcoef = xfdj.coefficients
            xbasis = xfdj.basis
            Jpsithetaj = xbasis.inner_product(beta[j])
            Zmat = xcoef @ Jpsithetaj if j == 0 else np.concatenate(
                (Zmat, xcoef @ Jpsithetaj), axis=1)

        if any(w != 1 for w in wt):
            rtwt = np.sqrt(wt)
            Zmatwt = Zmat * rtwt
            ymatwt = y * rtwt
            Cmat = np.transpose(Zmatwt @ Zmatwt + Rmat)
            Dmat = np.transpose(Zmatwt) @ ymatwt
        else:
            Cmat = np.transpose(Zmat) @ Zmat
            Dmat = np.transpose(Zmat) @ y

        # eigchk(Cmat)
        Cmatinv = np.linalg.inv(Cmat)
        betacoef = Cmatinv @ Dmat

        df = np.sum(np.diag(Zmat @ Cmatinv @ np.transpose(Zmat)))

        mj2 = 0
        for j in range(0, nbeta):
            mj1 = mj2
            mj2 = mj2 + beta[j].nbasis
            beta[j] = FDataBasis(beta[j], np.transpose(betacoef[mj1:mj2]))

        self.beta = beta

    def predict(self, x):
        return [sum(self.beta[i].inner_product(x[i][j])[0, 0] for i in
                    range(len(self.beta))) for j in range(x[0].nsamples)]

    def mean_squared_error(self, y_actual, y_predicted):
        return np.sqrt(mean_squared_error(y_actual, y_predicted))

    def _argcheck(self, y, x):
        """Do some checks to types and shapes"""
        if all(not isinstance(i, FDataBasis) for i in x):
            raise ValueError("All the dependent variable are scalar.")
        if any(isinstance(i, FDataBasis) for i in y):
            raise ValueError(
                "Some of the independent variables are not scalar")

        ylen = len(y)
        xlen = len(x)
        blen = len(self.beta)
        domain_range = ([i for i in x if isinstance(i, FDataBasis)][0]
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
