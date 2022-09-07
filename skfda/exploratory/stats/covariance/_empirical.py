from __future__ import annotations

from typing import Generic, TypeVar

from ...._utils._sklearn_adapter import BaseEstimator
from ....representation import FData

Input = TypeVar("Input", bound=FData)


class EmpiricalCovariance(
    BaseEstimator,
    Generic[Input],
):

    def __init__(
        self,
        *,
        assume_centered: bool = False,
    ) -> None:
        self.assume_centered = assume_centered

    def _fit_mean(self, X: Input):
        self.location_ = X.mean()
        if self.assume_centered:
            self.location_ *= 0

    def fit(self, X: Input, y: object = None) -> EmpiricalCovariance[Input]:

        self._fit_mean(X)

        self.covariance_ = X.cov()

        return self

    def score(self, X_test: Input, y: object = None) -> float:

        pass
