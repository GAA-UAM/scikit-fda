import pytest

# https://github.com/scikit-learn/scikit-learn/issues/8959
import numpy as np
try:
    np.set_printoptions(sign=' ')
except TypeError:
    pass

collect_ignore = ['setup.py']

pytest.register_assert_rewrite("skfda")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
