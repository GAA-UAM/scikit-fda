from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "constants",
    ],
    submod_attrs={
        "_utils": [
            "_cartesian_product",
            "_check_array_key",
            "_check_estimator",
            "_classifier_get_classes",
            "_evaluate_grid",
            "_int_to_real",
            "_MapAcceptable",
            "_pairwise_symmetric",
            "_same_domain",
            "_to_grid",
            "_to_grid_points",
            "nquad_vec",
        ],
        '_warping': [
            "invert_warping",
            "normalize_scale",
            "normalize_warping",
        ],
    },
)

if TYPE_CHECKING:

    from ._utils import (
        _cartesian_product as _cartesian_product,
        _check_array_key as _check_array_key,
        _check_estimator as _check_estimator,
        _classifier_get_classes as _classifier_get_classes,
        _evaluate_grid as _evaluate_grid,
        _int_to_real as _int_to_real,
        _MapAcceptable as _MapAcceptable,
        _pairwise_symmetric as _pairwise_symmetric,
        _same_domain as _same_domain,
        _to_grid as _to_grid,
        _to_grid_points as _to_grid_points,
        nquad_vec as nquad_vec,
    )

    from ._warping import (
        invert_warping as invert_warping,
        normalize_scale as normalize_scale,
        normalize_warping as normalize_warping,
    )
