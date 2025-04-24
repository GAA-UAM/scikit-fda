"""Imputation of missing values."""
from __future__ import annotations

import importlib  # noqa: F401
from typing import TYPE_CHECKING, Any  # noqa: F401

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_interpolate": ["MissingValuesInterpolation"],
    },
)

if TYPE_CHECKING:
    from ._interpolate import (  # noqa: I001
        MissingValuesInterpolation as MissingValuesInterpolation  # noqa: COM812
    )
