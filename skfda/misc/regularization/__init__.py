from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_regularization": [
            "L2Regularization",
            "TikhonovRegularization",
            "compute_penalty_matrix",
        ],

    },
)

if TYPE_CHECKING:
    from ._regularization import (
        L2Regularization as L2Regularization,
        TikhonovRegularization as TikhonovRegularization,
        compute_penalty_matrix as compute_penalty_matrix,
    )
