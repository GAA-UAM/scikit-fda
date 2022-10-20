from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, WhiteKernel

from ....misc.covariances import Covariance
from ....representation import FDataGrid
from ._empirical import EmpiricalCovariance


class ParametricGaussianCovariance(EmpiricalCovariance[FDataGrid]):

    def __init__(
        self,
        cov: Kernel | Covariance,
        *,
        assume_centered: bool = False,
        fit_noise: bool = True,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.cov = cov
        self.fit_noise = fit_noise

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> ParametricGaussianCovariance:

        self._fit_mean(X)

        X_centered = X - self.location_

        data_matrix = X_centered.data_matrix[:, :, 0]

        grid_points = X_centered.grid_points[0][:, np.newaxis]

        cov = self.cov
        to_sklearn = getattr(cov, "to_sklearn", None)
        if to_sklearn:
            cov = to_sklearn()

        if self.fit_noise:
            cov += WhiteKernel()

        regressor = GaussianProcessRegressor(kernel=cov)
        regressor.fit(grid_points, data_matrix.T)

        # TODO: Skip cov computation?
        self.covariance_ = X.cov().copy(
            data_matrix=regressor.kernel_(grid_points)[np.newaxis, ...],
        )

        return self
