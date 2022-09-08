"""Covariance estimation."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_base": ["CovarianceEstimator"],
        "_empirical": ["EmpiricalCovariance"],
        "_parametric_gaussian": ["ParametricGaussianCovariance"],
    },
)

if TYPE_CHECKING:
    from ._base import CovarianceEstimator as CovarianceEstimator
    from ._empirical import EmpiricalCovariance as EmpiricalCovariance
    from ._parametric_gaussian import (
        ParametricGaussianCovariance as ParametricGaussianCovariance,
    )
