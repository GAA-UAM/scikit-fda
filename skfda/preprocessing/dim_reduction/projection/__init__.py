import warnings  # noqa: D104

from .. import FPCA  # noqa: F401

warnings.warn(
    'The module "projection" is deprecated. Please use "dim_reduction"',
    category=DeprecationWarning,
    stacklevel=2,
)
