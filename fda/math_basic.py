"""This module defines the basic mathematic operations for classes defined in this package.

"""
from fda.FDataGrid import FDataGrid
import numpy
import scipy.stats.mstats


__author__ = "Miguel Carbajo Berrocal"
__license__ = "GPL3"
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def mean(fdatagrid):
    return FDataGrid([numpy.mean(fdatagrid.data_matrix, 0)], fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def var(fdatagrid):
    return FDataGrid([numpy.var(fdatagrid.data_matrix, 0)], fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def gmean(fdatagrid):
    return FDataGrid([scipy.stats.mstats.gmean(fdatagrid.data_matrix, 0)], fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def cov(fdatagrid):
    return numpy.cov(fdatagrid.data_matrix)


def sqrt(fdatagrid):
    return FDataGrid(numpy.sqrt(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def absolute(fdatagrid):
    return FDataGrid(numpy.absolute(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def around(fdatagrid, decimals=0):
    return FDataGrid(numpy.around(fdatagrid.data_matrix, decimals), fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def exp(fdatagrid):
    return FDataGrid(numpy.exp(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def log(fdatagrid):
    return FDataGrid(numpy.log(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def log10(fdatagrid):
    return FDataGrid(numpy.log10(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def log2(fdatagrid):
    return FDataGrid(numpy.log2(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)


def cumsum(fdatagrid):
    return FDataGrid(numpy.cumsum(fdatagrid.data_matrix), fdatagrid.sample_points, fdatagrid.sample_range, fdatagrid.names)
