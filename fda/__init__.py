import errno as _errno
from fda.basis import FDataBasis
from fda.grid import FDataGrid
from fda.math_basic import mean, var, gmean, log, log2, log10, exp, sqrt, \
    cumsum, metric, norm_lp, inner_product, cov
import os as _os


try:
    with open(_os.path.join(_os.path.dirname(__file__),
                            '..', 'VERSION'), 'r') as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"
