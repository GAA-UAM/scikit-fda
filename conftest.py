"""Configuration file for pytest."""
import asyncio
import contextlib
import importlib.util
import sys

import numpy as np
import pytest

# https://github.com/scikit-learn/scikit-learn/issues/8959

with contextlib.suppress(TypeError):
    np.set_printoptions(sign=" ")


# I introduced this change in order to adapt it to Windows 10
# operating system.
# More information about this problem in this GitHub issue:
# https://github.com/jupyter/jupyter-sphinx/issues/171#issuecomment-766953182

if (
    sys.version_info[0] >= 3 and sys.version_info[1] >= 8  # noqa: PLR2004, YTT203
        and sys.platform.startswith("win")
):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

collect_ignore = ["setup.py", "docs/conf.py", "asv_benchmarks"]

# PyTorch is an optional dependency, required only by skfda.ml.generative.
# When it is absent, --doctest-modules cannot import the generative source
# modules and the diffusion tests cannot import torch, so skip both dirs.
if importlib.util.find_spec("torch") is None:
    collect_ignore += ["skfda/ml/generative", "skfda/tests/generative"]

pytest.register_assert_rewrite("skfda")
