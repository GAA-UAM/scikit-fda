import warnings

from ..feature_extraction import FPCA

warnings.warn(
    'The module "projection" is deprecated. Please use "feature_extraction"',
    category=DeprecationWarning,
)
