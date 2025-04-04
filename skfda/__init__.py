"""scikit-fda package."""
import errno as _errno
import os as _os
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "datasets",
        "exploratory",
        "inference",
        "misc",
        "ml",
        "preprocessing",
        "representation",
    ],
    submod_attrs={
        'representation': [
            "FData", "FDataBasis", "FDataGrid", "FDataIrregular",
        ],
        'representation._functional_data': ['concatenate'],
    },
)

if TYPE_CHECKING:
    from .representation import (
        FData as FData,
        FDataBasis as FDataBasis,
        FDataGrid as FDataGrid,
        FDataIrregular as FDataIrregular,
        concatenate as concatenate,
    )

__version__ = "0.10.1"
