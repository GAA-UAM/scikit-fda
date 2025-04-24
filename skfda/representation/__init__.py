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
        '_functional_data': ["FData", "concatenate"],  # noqa: Q000
        'basis': ["FDataBasis"],  # noqa: Q000
        'grid': ["FDataGrid"],  # noqa: Q000
        'irregular': ["FDataIrregular"],  # noqa: Q000
    },
)

if TYPE_CHECKING:
    from ._functional_data import FData as FData, concatenate as concatenate
    from .basis import FDataBasis as FDataBasis
    from .grid import FDataGrid as FDataGrid
    from .irregular import FDataIrregular as FDataIrregular
