from __future__ import annotations

from typing import Callable, TypeVar

from ....representation import FData
from ....typing._numpy import NDArrayFloat
from ._base import CovarianceEstimator

Input = TypeVar("Input", bound=FData)


class EmpiricalCovariance(
    CovarianceEstimator[Input],
):
    covariance_: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]

    def fit(self, X: Input, y: object = None) -> EmpiricalCovariance[Input]:
        super().fit(X, y)
        self.covariance_ = X.cov()
        return self
