"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""
from typing import TYPE_CHECKING

import lazy_loader as lazy

# This cannot be made lazy for now
from ..._utils import invert_warping, normalize_warping

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
    from ._fisher_rao import ElasticRegistration, FisherRaoElasticRegistration
    from ._landmark_registration import (
        landmark_elastic_registration,
        landmark_elastic_registration_warping,
        landmark_registration,
        landmark_shift,
        landmark_shift_deltas,
        landmark_shift_registration,
    )
    from ._lstsq_shift_registration import (
        LeastSquaresShiftRegistration,
        ShiftRegistration,
    )
