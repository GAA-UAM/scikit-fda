from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from ...._utils._sklearn_adapter import BaseEstimator
from ....representation import FData

Input = TypeVar("Input", bound=FData)


class CovarianceEstimator(
    BaseEstimator,
    Generic[Input],
):

    location_: Input
    covariance_: Input

    def __init__(
        self,
        *,
        assume_centered: bool = False,
    ) -> None:
        self.assume_centered = assume_centered

    def _fit_mean(self, X: Input) -> None:
        self.location_ = X.mean()
        if self.assume_centered:
            self.location_ *= 0

    @abstractmethod
    def fit(self, X: Input, y: object = None) -> CovarianceEstimator[Input]:

        self._fit_mean(X)

        return self

    def score(self, X_test: Input, y: object = None) -> float:

        pass
