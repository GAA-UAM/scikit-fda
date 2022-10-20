"""Depth."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "multivariate",
    ],
    submod_attrs={
        '_depth': [
            "BandDepth",
            "DistanceBasedDepth",
            "IntegratedDepth",
            "ModifiedBandDepth",
        ],
        "multivariate": [
            "Depth",
            "Outlyingness",
            "OutlyingnessBasedDepth",
        ],
    },
)

if TYPE_CHECKING:
    from ._depth import (
        BandDepth as BandDepth,
        DistanceBasedDepth as DistanceBasedDepth,
        IntegratedDepth as IntegratedDepth,
        ModifiedBandDepth as ModifiedBandDepth,
    )
    from .multivariate import (
        Depth as Depth,
        Outlyingness as Outlyingness,
        OutlyingnessBasedDepth as OutlyingnessBasedDepth,
    )
