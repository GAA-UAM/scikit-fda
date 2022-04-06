import warnings

from ... import feature_construction

warnings.warn(
    'The module "feature_extraction" is deprecated.'
    'Please use "feature_construction"',
    category=DeprecationWarning,
)
