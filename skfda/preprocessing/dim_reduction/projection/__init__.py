import warnings

from .. import FPCA

warnings.warn(
    'The module "projection" is deprecated. Please use "dim_reduction"',
    category=DeprecationWarning,
)
