"""This module defines the basic mathematic operations for classes defined
in this package.

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
    """ Computes the mean of all the samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the samples whose mean
            is wanted.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        mean of all the samples in the original FDataGrid object.

    """
    return FDataGrid([numpy.mean(fdatagrid.data_matrix, 0)],
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def var(fdatagrid):
    """ Computes the variance of a set of samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the set of samples
        whose variance is desired.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        mean of all the samples in the original FDataGrid object.

    """
    return FDataGrid([numpy.var(fdatagrid.data_matrix, 0)],
                     fdatagrid.sample_points,  fdatagrid.sample_range,
                     fdatagrid.names)


def gmean(fdatagrid):
    """ Computes the geometric mean of all the samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the samples whose
            geometric mean is wanted.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        geometric mean of all the samples in the original FDataGrid object.

    """
    return FDataGrid([scipy.stats.mstats.gmean(fdatagrid.data_matrix, 0)],
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def cov(fdatagrid):
    """ Calculates the covariance matrix representing the covariance of the
    functional samples at the observation points.

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
        functional variable.

    Returns:
        numpy.darray: Matrix of covariances.

    """
    return numpy.cov(fdatagrid.data_matrix)


def sqrt(fdatagrid):
    """ Performs a element wise square root operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the square root
            operations is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the square roots of the original.

    """
    return FDataGrid(numpy.sqrt(fdatagrid.data_matrix),
                     fdatagrid.sample_points,  fdatagrid.sample_range,
                     fdatagrid.names)


def absolute(fdatagrid):
    """ Gets the absolute value of all elements in the FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object from whose elements the absolute value
            is going to be retrieved.

    Returns:
        FDataGrid: Object whose elements are the absolute values of the
            original.

    """
    return FDataGrid(numpy.absolute(fdatagrid.data_matrix),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def around(fdatagrid, decimals=0):
    """ Rounds all elements of the object.

    Args:
        fdatagrid (FDataGrid): Object to whose elements are going to be
            rounded.
        decimals (int, optional): Number of decimals wanted. Defaults to 0.

    Returns:
        FDataGrid: Object whose elements are rounded.

    """
    return FDataGrid(numpy.around(fdatagrid.data_matrix, decimals),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def exp(fdatagrid):
    """ Performs a element wise exponential operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the exponential
            operations is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the result of exponentiating
            the elements of the original.

    """
    return FDataGrid(numpy.exp(fdatagrid.data_matrix),
                     fdatagrid.sample_points,  fdatagrid.sample_range,
                     fdatagrid.names)


def log(fdatagrid):
    """ Performs a element wise logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the logarithm
            operations is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the logarithm of the original.

    """
    return FDataGrid(numpy.log(fdatagrid.data_matrix),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def log10(fdatagrid):
    """ Performs a element wise base 10 logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the base 10 logarithm
            operations is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the base 10 logarithm of the
            original.

    """
    return FDataGrid(numpy.log10(fdatagrid.data_matrix),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def log2(fdatagrid):
    """ Performs a element wise binary logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the binary logarithm
            operations is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the binary logarithm of the
            original.

    """
    return FDataGrid(numpy.log2(fdatagrid.data_matrix),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)


def cumsum(fdatagrid):
    """ Returns the cumulative sum of the samples.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the cumulative sum is
            going to be calculated.

    Returns:
        FDataGrid: Object with the sample wise cumulative sum.

    """
    return FDataGrid(numpy.cumsum(fdatagrid.data_matrix, axis=0),
                     fdatagrid.sample_points, fdatagrid.sample_range,
                     fdatagrid.names)
