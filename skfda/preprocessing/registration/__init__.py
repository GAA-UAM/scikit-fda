"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""
from typing import TYPE_CHECKING

import lazy_loader as lazy

from ..._utils import (
    invert_warping as invert_warping,
    normalize_warping as normalize_warping,
)

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "validation",
    ],
    submod_attrs={
        "_fisher_rao": ["ElasticRegistration", "FisherRaoElasticRegistration"],
        "_landmark_registration": [
            "landmark_elastic_registration",
            "landmark_elastic_registration_warping",
            "landmark_registration",
            "landmark_shift",
            "landmark_shift_deltas",
            "landmark_shift_registration",
        ],
        "_lstsq_shift_registration": [
            "LeastSquaresShiftRegistration",
            "ShiftRegistration",
        ],
    },
)

if TYPE_CHECKING:
    from ._fisher_rao import (
        ElasticRegistration as ElasticRegistration,
        FisherRaoElasticRegistration as FisherRaoElasticRegistration,
    )
    from ._landmark_registration import (
        landmark_elastic_registration as landmark_elastic_registration,
        landmark_elastic_registration_warping as landmark_elastic_registration_warping,
        landmark_registration as landmark_registration,
        landmark_shift as landmark_shift,
        landmark_shift_deltas as landmark_shift_deltas,
        landmark_shift_registration as landmark_shift_registration,
    )
    from ._lstsq_shift_registration import (
        LeastSquaresShiftRegistration as LeastSquaresShiftRegistration,
        ShiftRegistration as ShiftRegistration,
    )
