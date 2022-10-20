"""Regression."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_historical_linear_model": ["HistoricalLinearRegression"],
        "_kernel_regression": ["KernelRegression"],
        "_linear_regression": ["LinearRegression"],
        "_neighbors_regression": [
            "KNeighborsRegressor",
            "RadiusNeighborsRegressor",
        ],
    },
)

if TYPE_CHECKING:
    from ._historical_linear_model import (
        HistoricalLinearRegression as HistoricalLinearRegression,
    )
    from ._kernel_regression import KernelRegression as KernelRegression
    from ._linear_regression import LinearRegression as LinearRegression
    from ._neighbors_regression import (
        KNeighborsRegressor as KNeighborsRegressor,
        RadiusNeighborsRegressor as RadiusNeighborsRegressor,
    )
