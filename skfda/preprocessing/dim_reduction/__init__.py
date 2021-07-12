"""Dim reduction."""
from __future__ import annotations

import importlib
from typing import Any

from . import feature_extraction, variable_selection


def __getattr__(name: str) -> Any:
    if name == "projection":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
