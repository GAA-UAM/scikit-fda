"""Feature extraction."""
import warnings

from .. import FPCA  # noqa: F401

warnings.warn(
    'The module "feature_extraction" is deprecated.'
    'Please use "dim_reduction" for FPCA'
    'or "feature_construction" for feature construction techniques',
    category=DeprecationWarning,
    stacklevel=2,
)
