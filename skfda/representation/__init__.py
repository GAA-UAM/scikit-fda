"""Representation of functional data."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "basis",
        "extrapolation",
        "grid",
        "interpolation",
        "irregular",
    ],
    submod_attrs={
        '_functional_data': ["FData", "concatenate"],
        'basis': ["FDataBasis"],
        'grid': ["FDataGrid"],
        'irregular': ["FDataIrregular"],
    },
)

if TYPE_CHECKING:
    from ._functional_data import FData as FData, concatenate as concatenate
    from .basis import FDataBasis as FDataBasis
    from .grid import FDataGrid as FDataGrid
    from .irregular import FDataIrregular as FDataIrregular
