"""Conversion."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_mixed_effects": [
            "EMMixedEffectsConverter",
            "MixedEffectsConverter",
            "MinimizeMixedEffectsConverter"
        ],
    },
)

if TYPE_CHECKING:
    from ._mixed_effects import (
        EMMixedEffectsConverter,
        MinimizeMixedEffectsConverter,
        MixedEffectsConverter,
    )
