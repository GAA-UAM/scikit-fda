from __future__ import annotations

from typing import Callable, TypeVar

from ....misc.covariances import Empirical
from ....representation import FData
from ....typing._numpy import NDArrayFloat
from ._base import CovarianceEstimator

Input = TypeVar("Input", bound=FData)


class EmpiricalCovariance(
    CovarianceEstimator[Input],
):
    """Empirical covariance estimator."""

    covariance_: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]

    def fit(self, X: Input, y: object = None) -> EmpiricalCovariance[Input]:
        """Fit the covariance estimator."""
        super().fit(X, y)
        self.covariance_ = X.cov()
        assert isinstance(self.covariance_, Empirical)
        # TODO: Use a property setter for instance
        self.covariance_.set_regularization_parameter(
            self.regularization_parameter,
        )
        return self

    def score(
        self,
        X_test: Input,
        y: object = None,
    ) -> NDArrayFloat:
        """Compute the log-likelihood of the data."""
        assert isinstance(self.covariance_, Empirical)
        log_determinant = self.covariance_.log_determinant()
        mahalanobis = self.mahalanobis(
            X_test,
        )
        return -0.5 * (mahalanobis + log_determinant)
