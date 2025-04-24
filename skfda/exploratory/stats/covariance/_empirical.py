from __future__ import annotations

from typing import Callable, TypeVar  # noqa: UP035

from ....representation import FData
from ....typing._numpy import NDArrayFloat  # noqa: TC001
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
