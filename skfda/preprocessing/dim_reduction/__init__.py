"""Dim reduction."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import lazy_loader as lazy

_normal_getattr, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "variable_selection",
    ],
    submod_attrs={
        "_fpca": ["FPCA"],
        "_neighbor_transforms": ["KNeighborsTransformer"]
    },
)

if TYPE_CHECKING:
    from ._fpca import FPCA as FPCA
    from ._neighbor_transforms import KNeighborsTransformer as KNeighborsTransformer


def __getattr__(name: str) -> Any:
    if name in {"projection", "feature_extraction"}:
        return importlib.import_module(f".{name}", __name__)
    return _normal_getattr(name)
