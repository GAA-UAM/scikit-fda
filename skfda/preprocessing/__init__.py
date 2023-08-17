"""Preprocessing methods for functional data."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "dim_reduction",
        "feature_construction",
        "missing",
        "registration",
        "smoothing",
    ],
)
