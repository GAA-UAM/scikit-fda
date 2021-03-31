"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""

from . import elastic, validation
from ._landmark_registration import (
    landmark_registration,
    landmark_registration_warping,
    landmark_shift,
    landmark_shift_deltas,
)
from ._shift_registration import ShiftRegistration
from ._warping import invert_warping, normalize_warping
from .elastic import ElasticRegistration
