"""Feature extraction."""
import warnings

from .. import FPCA

warnings.warn(
    'The module "feature_extraction" is deprecated.'
    'Please use "dim_reduction" for FPCA'
    'or "feature_construction" for feature construction techniques',
    category=DeprecationWarning,
)
