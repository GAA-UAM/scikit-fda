"""Basic mathematical functionalities of the package.

Defines the basic mathematical operations for classes defined in this
package. FDataBasis and FDataGrid.

"""
import numpy

import scipy.stats.mstats

from . import FDataGrid


__author__ = "Miguel Carbajo Berrocal"
__license__ = "GPL3"
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def mean(fdata):
    """Compute the mean of all the samples in a FData object.

    Computes the mean of all the samples in a FDataGrid or FDataBasis object.

    Args:
        fdata(FDataGrid or FDataBasis): Object containing all the samples
        whose mean
            is wanted.

    Returns:
        FDataGrid or FDataBasis: A FDataGrid or FDataBasis object with just
        one sample representing the mean of all the samples in the original
        object.

    """
    return fdata.mean()


def var(fdatagrid):
    """Compute the variance of a set of samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the set of samples
        whose variance is desired.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        variance of all the samples in the original FDataGrid object.

    """
    return fdatagrid.var()


def gmean(fdatagrid):
    """Compute the geometric mean of all the samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the samples whose
            geometric mean is wanted.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        geometric mean of all the samples in the original FDataGrid object.

    """
    return fdatagrid.gmean()


def cov(fdatagrid):
    """Compute the covariance.

    Calculates the covariance matrix representing the covariance of the
    functional samples at the observation points.

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.

    Returns:
        numpy.darray: Matrix of covariances.

    """
    return fdatagrid.cov()


def sqrt(fdatagrid):
    """Perform a element wise square root operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the square root
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the square roots of the original.

    """
    return fdatagrid.copy(data_matrix=numpy.sqrt(fdatagrid.data_matrix))


def absolute(fdatagrid):
    """Get the absolute value of all elements in the FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object from whose elements the absolute value
            is going to be retrieved.

    Returns:
        FDataGrid: Object whose elements are the absolute values of the
            original.

    """
    return fdatagrid.copy(data_matrix=numpy.absolute(fdatagrid.data_matrix))


def round(fdatagrid, decimals=0):
    """Round all elements of the object.

    Args:
        fdatagrid (FDataGrid): Object to whose elements are going to be
            rounded.
        decimals (int, optional): Number of decimals wanted. Defaults to 0.

    Returns:
        FDataGrid: Object whose elements are rounded.

    """
    return fdatagrid.round(decimals)


def exp(fdatagrid):
    """Perform a element wise exponential operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the exponential
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the result of exponentiating
            the elements of the original.

    """
    return fdatagrid.copy(data_matrix=numpy.exp(fdatagrid.data_matrix))


def log(fdatagrid):
    """Perform a element wise logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the logarithm of the original.

    """
    return fdatagrid.copy(data_matrix=numpy.log(fdatagrid.data_matrix))


def log10(fdatagrid):
    """Perform an element wise base 10 logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the base 10 logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the base 10 logarithm of the
            original.

    """
    return fdatagrid.copy(data_matrix=numpy.log10(fdatagrid.data_matrix))


def log2(fdatagrid):
    """Perform an element wise binary logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the binary logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the binary logarithm of the
            original.

    """
    return fdatagrid.copy(data_matrix=numpy.log2(fdatagrid.data_matrix))


def cumsum(fdatagrid):
    """Return the cumulative sum of the samples.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the cumulative sum is
            going to be calculated.

    Returns:
        FDataGrid: Object with the sample wise cumulative sum.

    """
    return fdatagrid.copy(data_matrix=numpy.cumsum(fdatagrid.data_matrix,
                                                   axis=0))


def inner_product(fdatagrid, fdatagrid2):
    r"""Return inner product for FDataGrid.

    Calculates the inner product amongst all the samples in two
    FDataGrid objects.

    For each pair of samples f and g the inner product is defined as:

    .. math::
        <f, g> = \int_a^bf(x)g(x)dx

    The integral is approximated using Simpson's rule.

    Args:
        fdatagrid (FDataGrid): First FDataGrid object.
        fdatagrid2 (FDataGrid): Second FDataGrid object.

    Returns:
        numpy.darray: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the inner product of the ith sample
        of the first object and the jth sample of the second one.

    Examples:
        The inner product of the :math:'f(x) = x` and the constant
        :math:`y=1` defined over the interval [0,1] is the area of the
        triangle delimited by the the lines y = 0, x = 1 and y = x; 0.5.

        >>> x = numpy.linspace(0,1,1001)
        >>> fd1 = FDataGrid(x,x)
        >>> fd2 = FDataGrid(numpy.ones(len(x)),x)
        >>> inner_product(fd1, fd2)
        array([[ 0.5]])

        If the FDataGrid object contains more than one sample

        >>> fd1 = FDataGrid([x, numpy.ones(len(x))], x)
        >>> fd2 = FDataGrid([numpy.ones(len(x)), x] ,x)
        >>> inner_product(fd1, fd2).round(2)
        array([[ 0.5 , 0.33],
               [ 1.  , 0.5 ]])

    """
    if fdatagrid.ndim_domain != 1:
        raise NotImplementedError("This method only works when the dimension "
                                  "of the domain of the FDatagrid object is "
                                  "one.")
    # Checks
    if not numpy.array_equal(fdatagrid.sample_points,
                             fdatagrid2.sample_points):
        raise ValueError("Sample points for both objects must be equal")

    # Creates an empty matrix with the desired size to store the results.
    matrix = numpy.empty([fdatagrid.nsamples, fdatagrid2.nsamples])
    # Iterates over the different samples of both objects.
    for i in range(fdatagrid.nsamples):
        for j in range(fdatagrid2.nsamples):
            # Calculates the inner product using Simpson's rule.
            matrix[i, j] = (scipy.integrate.simps(
                fdatagrid.data_matrix[i, ..., 0] *
                fdatagrid2.data_matrix[j, ..., 0],
                x=fdatagrid.sample_points[0]
                ))
    return matrix


def fpca(fdatagrid, n=2):
    """Compute Functional Principal Components Analysis.

    Performs Functional Principal Components Analysis to reduce
    dimensionality and obtain the principal modes of variation for a
    functional data object.

    It uses SVD numpy implementation to compute PCA.

    Args:
        fdatagrid (FDataGrid): functional data object.
        n (int, optional): Number of principal components. Defaults to 2.

    Returns:
        tuple: (scores, principal directions, eigenvalues)

    """
    fdatagrid = fdatagrid - mean(fdatagrid)  # centers the data
    # singular value decomposition
    u, s, v = numpy.linalg.svd(fdatagrid.data_matrix)
    principal_directions = v.T  # obtain the eigenvectors matrix
    eigenvalues = (numpy.diag(s) ** 2) / (fdatagrid.nsamples - 1)
    scores = u @ s  # functional principal scores

    return scores, principal_directions, eigenvalues
