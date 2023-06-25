from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, WhiteKernel

from ....misc.covariances import Covariance, EmpiricalGrid
from ....representation import FDataGrid
from ....typing._numpy import NDArrayFloat
from ._empirical import EmpiricalCovariance


class ParametricGaussianCovariance(EmpiricalCovariance[FDataGrid]):
    """Parametric Gaussian covariance estimator."""

    covariance_: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]

    def __init__(
        self,
        cov: Kernel | Covariance,
        *,
        assume_centered: bool = False,
        regularization_parameter: float = 0,
        fit_noise: bool = True,
    ) -> None:
        super().__init__(
            assume_centered=assume_centered,
            regularization_parameter=regularization_parameter,
        )
        self.cov = cov
        self.fit_noise = fit_noise

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> ParametricGaussianCovariance:
        """Fit the covariance estimator."""
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
        # TODO: Use a user-public structure to represent the covariance,
        #  instead of a Callable object
        self.covariance_ = X.cov()
        assert isinstance(self.covariance_, EmpiricalGrid)
        self.covariance_.cov_fdata = self.covariance_.cov_fdata.copy(
            data_matrix=regressor.kernel_(
                grid_points,
            )[np.newaxis, ...],
        )
        self.covariance_.set_regularization_parameter(
            self.regularization_parameter,
        )

        return self

    def score(
        self,
        X_test: FDataGrid,
        y: object = None,
    ) -> NDArrayFloat:
        """Compute the log-likelihood of the data."""
        assert isinstance(self.covariance_, EmpiricalGrid)
        log_determinant = self.covariance_.log_determinant()
        mahalanobis = self.mahalanobis(
            X_test,
        )
        return -0.5 * (mahalanobis + log_determinant)
