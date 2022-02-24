"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""

from ..._utils import invert_warping, normalize_warping
from . import validation
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
