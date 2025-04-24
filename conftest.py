import pytest  # noqa: I001, D100
import sys
import asyncio
import numpy as np

# https://github.com/scikit-learn/scikit-learn/issues/8959

try:  # noqa: SIM105
    np.set_printoptions(sign=' ')  # noqa: Q000
except TypeError:
    pass


# I introduced this change in order to adapt it to Windows 10
# operating system.
# More information about this problem in this GitHub issue:
# https://github.com/jupyter/jupyter-sphinx/issues/171#issuecomment-766953182

if (
    sys.version_info[0] == 3 and sys.version_info[1] >= 8  # noqa: YTT201, YTT203, PLR2004
        and sys.platform.startswith('win')  # noqa: Q000
):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

collect_ignore = ['setup.py', 'docs/conf.py', 'asv_benchmarks']  # noqa: Q000

pytest.register_assert_rewrite("skfda")
