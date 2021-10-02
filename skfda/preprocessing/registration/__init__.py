"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""

from ..._utils import invert_warping, normalize_warping
from . import validation
from ._fisher_rao import ElasticFisherRaoRegistration, ElasticRegistration
from ._landmark_registration import (
    landmark_registration,
    landmark_registration_warping,
    landmark_shift,
    landmark_shift_deltas,
)
from ._shift_registration import ShiftRegistration
