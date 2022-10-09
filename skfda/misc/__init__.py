"""Miscellaneous functions and objects."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "covariances",
        "hat_matrix",
        "kernels",
        "lstsq",
        "metrics",
        "operators",
        "regularization",
        "scoring",
        "validation",
    ],
    submod_attrs={
        '_math': [
            "cosine_similarity",
            "cosine_similarity_matrix",
            "cumsum",
            "exp",
            "inner_product",
            "inner_product_matrix",
            "log",
            "log2",
            "log10",
            "sqrt",
        ],
    },
)

if TYPE_CHECKING:

    from ._math import (
        cosine_similarity as cosine_similarity,
        cosine_similarity_matrix as cosine_similarity_matrix,
        cumsum as cumsum,
        exp as exp,
        inner_product as inner_product,
        inner_product_matrix as inner_product_matrix,
        log as log,
        log2 as log2,
        log10 as log10,
        sqrt as sqrt,
    )
