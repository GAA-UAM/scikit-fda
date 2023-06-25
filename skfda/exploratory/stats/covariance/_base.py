from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Generic, TypeVar

from ...._utils._sklearn_adapter import BaseEstimator
from ....misc.rkhs_product import rkhs_inner_product
from ....representation import FData
from ....typing._numpy import NDArrayFloat

Input = TypeVar("Input", bound=FData)


class CovarianceEstimator(
    BaseEstimator,
    Generic[Input],
):
    """Base class for covariance estimators."""

    location_: Input
    covariance_: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]
    regularization_parameter: float

    def __init__(
        self,
        *,
        assume_centered: bool = False,
        regularization_parameter: float = 0,
    ) -> None:
        self.assume_centered = assume_centered
        self.regularization_parameter = regularization_parameter

    def _fit_mean(self, X: Input) -> None:
        self.location_ = X.mean()
        if self.assume_centered:
            self.location_ *= 0

    @abstractmethod
    def fit(self, X: Input, y: object = None) -> CovarianceEstimator[Input]:
        """Fit the covariance estimator."""
        self._fit_mean(X)
        return self

    @abstractmethod
    def score(
        self,
        X_test: Input,
        y: object = None,
    ) -> NDArrayFloat:
        """Compute the log-likelihood of the data."""
        pass

    def mahalanobis(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """Compute the squared Mahalanobis distance of each sample.

        Args:
            X: Functional data to compute the distance to. It represents the
                (x-y) term in the Mahalanobis distance.

        Returns:
            Mahalanobis distance of each sample. This is
            a matrix of shape (n_samples).
        """
        return rkhs_inner_product(
            fdata1=X,
            fdata2=X,
            cov_function=self.covariance_,
        )
