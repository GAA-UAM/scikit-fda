"""Dim reduction."""
from __future__ import annotations

import importlib
from typing import Any

from . import variable_selection
from ._fpca import FPCA


def __getattr__(name: str) -> Any:
    if name in {"projection", "feature_extraction"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
