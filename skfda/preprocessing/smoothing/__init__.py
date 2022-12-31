"""Smoothing."""
import warnings
from typing import TYPE_CHECKING, Any

import lazy_loader as lazy

_normal_getattr, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "validation",
    ],
    submod_attrs={
        "_basis": ["BasisSmoother"],
        "_kernel_smoothers": ["KernelSmoother"],
    },
)

if TYPE_CHECKING:
    from ._basis import BasisSmoother as BasisSmoother
    from ._kernel_smoothers import KernelSmoother as KernelSmoother

__kernel_smoothers__imported__ = False


def __getattr__(name: str) -> Any:
    global __kernel_smoothers__imported__
    if name == "kernel_smoothers" and not __kernel_smoothers__imported__:
        __kernel_smoothers__imported__ = True
        from . import kernel_smoothers
        return kernel_smoothers

    return _normal_getattr(name)
