import pytest
import sys
import asyncio
import numpy as np

# https://github.com/scikit-learn/scikit-learn/issues/8959

try:
    np.set_printoptions(sign=' ')
except TypeError:
    pass


# I introduced this change in order to adapt it to Windows 10
# operating system.
# More information about this problem in this GitHub issue:
# https://github.com/jupyter/jupyter-sphinx/issues/171#issuecomment-766953182

if (
    sys.version_info[0] == 3 and sys.version_info[1] >= 8
        and sys.platform.startswith('win')
):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

collect_ignore = ['setup.py', 'docs/conf.py', 'asv_benchmarks']

pytest.register_assert_rewrite("skfda")
