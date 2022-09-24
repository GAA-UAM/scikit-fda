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
        'representation': ["FData", "FDataBasis", "FDataGrid"],
        'representation._functional_data': ['concatenate'],
    },
)

if TYPE_CHECKING:
    from .representation import (
        FData as FData,
        FDataBasis as FDataBasis,
        FDataGrid as FDataGrid,
        concatenate as concatenate,
    )

try:
    with open(
        _os.path.join(
            _os.path.dirname(__file__),
            '..',
            'VERSION',
        ),
        'r',
    ) as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"
