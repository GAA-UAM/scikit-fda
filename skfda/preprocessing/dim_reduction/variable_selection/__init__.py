"""Variable selection methods for functional data."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "maxima_hunting",
        "mrmr",
        "recursive_maxima_hunting",
    ],
    submod_attrs={
        "_rkvs": ["RKHSVariableSelection"],
        "maxima_hunting": ["MaximaHunting"],
        "mrmr": ["MinimumRedundancyMaximumRelevance"],
        "recursive_maxima_hunting": ["RecursiveMaximaHunting"],
    },
)

if TYPE_CHECKING:
    from ._rkvs import RKHSVariableSelection as RKHSVariableSelection
    from .maxima_hunting import MaximaHunting as MaximaHunting
    from .mrmr import (
        MinimumRedundancyMaximumRelevance as MinimumRedundancyMaximumRelevance,
    )
    from .recursive_maxima_hunting import (
        RecursiveMaximaHunting as RecursiveMaximaHunting,
    )
