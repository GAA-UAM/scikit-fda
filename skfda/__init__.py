"""scikit-fda package."""
import errno as _errno  # noqa: F401
import os as _os  # noqa: F401
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
        'representation': [  # noqa: Q000
            "FData", "FDataBasis", "FDataGrid", "FDataIrregular",
        ],
        'representation._functional_data': ['concatenate'],  # noqa: Q000
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

__version__ = "0.10.2.dev0"
