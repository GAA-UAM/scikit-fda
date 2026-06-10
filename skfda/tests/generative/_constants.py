"""Numeric constants shared across all diffusion process test files.

Defined once here so that _constants.py, diffusion_test_mixins.py, and
every test file stay in sync. Import with a relative import:

    from ._constants import BATCH_SIZE, DATA_DIM, CUSTOM_DIM, SEED

conftest.py is the only file that cannot use this module (pytest loads
it before the package import system is initialised). Its copy of these
values is hardcoded inline with a reference comment pointing here.
"""

BATCH_SIZE = 8   # N: batch size for all tests that need input data
DATA_DIM   = 16  # M: data dimension (grid size)
CUSTOM_DIM = 17  # Deliberately != DATA_DIM to catch dimension-mixing bugs
SEED       = 13  # Random seed for all tests that need random data