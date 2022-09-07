from __future__ import annotations

import numpy as np
from GPy.models import GPRegression

from ....representation import FDataGrid
from ._empirical import EmpiricalCovariance


class ParametricGaussianCovariance(EmpiricalCovariance[FDataGrid]):

    def __init__(
        self,
        cov,
        *,
        assume_centered: bool = False,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.cov = cov

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> ParametricGaussianCovariance:

        self._fit_mean(X)

        X_centered = X - self.location_

        data_matrix = X_centered.data_matrix[:, :, 0]

        grid_points = X_centered.grid_points[0][:, np.newaxis]

        regressor = GPRegression(grid_points, data_matrix.T, kernel=self.cov)
        regressor.optimize()

        # TODO: Skip cov computation?
        self.covariance_ = X.cov().copy(
            data_matrix=regressor.kern.K(grid_points)[np.newaxis, ...],
        )

        return self
