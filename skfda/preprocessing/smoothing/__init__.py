"""Smoothing."""
import warnings
from typing import Any

from . import validation
from ._basis import BasisSmoother
from ._kernel_smoothers import KernelSmoother

__kernel_smoothers__imported__ = False


def __getattr__(name: str) -> Any:
    global __kernel_smoothers__imported__
    if name == "kernel_smoothers" and not __kernel_smoothers__imported__:
        __kernel_smoothers__imported__ = True
        from . import kernel_smoothers
        return kernel_smoothers

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
