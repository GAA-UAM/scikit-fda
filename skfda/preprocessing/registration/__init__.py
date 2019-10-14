"""Registration of functional data module.

This module contains methods to perform the registration of
functional data, in basis as well in discretized form.
"""

from ._landmark_registration import (landmark_shift_deltas,
                                     landmark_shift,
                                     landmark_registration_warping,
                                     landmark_registration)

from ._shift_registration import ShiftRegistration

from ._registration_utils import (invert_warping,
                                  normalize_warping,
                                  _normalize_scale)

from . import validation

from ._elastic import (to_srsf, from_srsf,
                       elastic_registration,
                       elastic_registration_warping,
                       elastic_mean, warping_mean)
