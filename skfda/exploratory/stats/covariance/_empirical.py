from __future__ import annotations

from typing import Generic, TypeVar

from ....representation import FData
from ._base import CovarianceEstimator

Input = TypeVar("Input", bound=FData)


class EmpiricalCovariance(
    CovarianceEstimator[Input],
):

    def fit(self, X: Input, y: object = None) -> EmpiricalCovariance[Input]:

        super(EmpiricalCovariance, self).fit(X, y)

        self.covariance_ = X.cov()

        return self
