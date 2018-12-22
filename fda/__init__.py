"""fda package.

It includes the modules:
    - basis: For functional data manipulation in a function basis system.
    - grid: For functional data manipulation as a discrete set of measures.
    - math: Mean, variance, covariance, logarithms, square roots, distances...
    - kernels: kernels for kernel smoothing.
    - kernel_smoothers: kernel smoothers for smoothing FDataGrid objects.
    - validation: cross validation methods for finding the parameter that
    best smooths a FDataGrid object.
and the following classes:
    - FDataGrid: Discrete representation of functional data.
    - FDataBasis: Basis representation for functional data.

"""
import errno as _errno
from fda.basis import FDataBasis
from fda.grid import FDataGrid
from fda.math import mean, var, gmean, log, log2, log10, exp, sqrt, \
    cumsum, metric, norm_lp, inner_product, cov
import os as _os

from . import datasets, covariances, registration, extrapolation

try:
    with open(_os.path.join(_os.path.dirname(__file__),
                            '..', 'VERSION'), 'r') as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"
