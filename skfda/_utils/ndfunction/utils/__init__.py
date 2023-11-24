
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "validation",
    ],
    submod_attrs={
        "_points": [
            "cartesian_product",
            "grid_points_equal",
        ],
    },
)

if TYPE_CHECKING:
    from ._points import (
        cartesian_product as cartesian_product,
        grid_points_equal as grid_points_equal,
    )
