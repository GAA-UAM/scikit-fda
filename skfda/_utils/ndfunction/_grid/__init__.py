from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_grid": [
            "GridDiscretizedFunction",
        ],
    },
)

if TYPE_CHECKING:

    from ._grid import GridDiscretizedFunction
