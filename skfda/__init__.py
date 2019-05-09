"""fda package.

It includes the modules:
    - basis: For functional data manipulation in a function basis system.
    - grid: For functional data manipulation as a discrete set of measures.
    - math: Mean, variance, covariance, logarithms, square roots, distances...
    - kernels: kernels for kernel smoothing.
    - kernel_smoothers: kernel smoothers for smoothing FDataGrid objects.
    - validation: cross validation methods for finding the parameter that
    best smooths a FDataGrid object.
    - depth_measures: depth methods to order he samples of FDataGrid objects.
    - magnitude-shape plot: visualization tool.
    - fdata_boxplot: informative exploratory tool for visualizing functional data.
and the following classes:
    - FDataGrid: Discrete representation of functional data.
    - FDataBasis: Basis representation for functional data.
    - Boxplot: Implements the functional boxplot for FDataGrid with domain dimension 1.
    - SurfaceBoxplot: Implements the functional boxplot for FDataGrid with
    domain dimension 2.
    - MagnitudeShapePlot: Implements the magnitude shape plot for FDataGrid
    with domain dimension 1.

"""
import errno as _errno

from .core import *
from .representation import FData
from .representation import FDataBasis
from .representation import FDataGrid
from .math import mean, var, gmean, log, log2, log10, exp, sqrt, \
    cumsum, inner_product, cov
from .metrics import lp_distance, norm_lp


import os as _os

from . import representation, datasets, covariances, preprocessing, exploratory

try:
    with open(_os.path.join(_os.path.dirname(__file__),
                            '..', 'VERSION'), 'r') as version_file:
        __version__ = version_file.read().strip()
except IOError as e:
    if e.errno != _errno.ENOENT:
        raise

    __version__ = "0.0"
