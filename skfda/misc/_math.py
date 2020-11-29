"""Basic mathematical functionalities of the package.

Defines the basic mathematical operations for classes defined in this
package. FDataBasis and FDataGrid.

"""
from builtins import isinstance
from typing import Union

import multimethod
import scipy.integrate

import numpy as np

from .._utils import _same_domain, nquad_vec, _pairwise_commutative
from ..representation import FDataGrid, FDataBasis
from ..representation.basis import Basis


__author__ = "Miguel Carbajo Berrocal"
__license__ = "GPL3"
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def sqrt(fdatagrid):
    """Perform a element wise square root operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the square root
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the square roots of the original.

    """
    return fdatagrid.copy(data_matrix=np.sqrt(fdatagrid.data_matrix))


def absolute(fdatagrid):
    """Get the absolute value of all elements in the FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object from whose elements the absolute value
            is going to be retrieved.

    Returns:
        FDataGrid: Object whose elements are the absolute values of the
            original.

    """
    return fdatagrid.copy(data_matrix=np.absolute(fdatagrid.data_matrix))


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
    return fdatagrid.copy(data_matrix=np.exp(fdatagrid.data_matrix))


def log(fdatagrid):
    """Perform a element wise logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the logarithm of the original.

    """
    return fdatagrid.copy(data_matrix=np.log(fdatagrid.data_matrix))


def log10(fdatagrid):
    """Perform an element wise base 10 logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the base 10 logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the base 10 logarithm of the
            original.

    """
    return fdatagrid.copy(data_matrix=np.log10(fdatagrid.data_matrix))


def log2(fdatagrid):
    """Perform an element wise binary logarithm operation.

    Args:
        fdatagrid (FDataGrid): Object to whose elements the binary logarithm
            operation is going to be applied.

    Returns:
        FDataGrid: Object whose elements are the binary logarithm of the
            original.

    """
    return fdatagrid.copy(data_matrix=np.log2(fdatagrid.data_matrix))


def cumsum(fdatagrid):
    """Return the cumulative sum of the samples.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the cumulative sum is
            going to be calculated.

    Returns:
        FDataGrid: Object with the sample wise cumulative sum.

    """
    return fdatagrid.copy(data_matrix=np.cumsum(fdatagrid.data_matrix,
                                                axis=0))


@multimethod.multidispatch
def inner_product(arg1, arg2, *, matrix=False, **kwargs):
    r"""Return the usual (:math:`L_2`) inner product.

    Calculates the inner product between matching samples in two
    FDataGrid objects.

    For two samples x and y the inner product is defined as:

    .. math::
        <x, y> = \sum_i x_i y_i

    for multivariate data and

    .. math::
        <x, y> = \int_a^b x(t)y(t)dt

    for functional data.

    The two arguments must have the same number of samples, or one should
    contain only one sample (and will be broadcasted).

    Args:

        arg1: First sample.
        arg2: Second sample.

    Returns:

        numpy.darray: Vector with the inner products of each pair of
            samples.

    Examples:

        This function can compute the multivariate inner product.

        >>> import numpy as np
        >>> from skfda.misc import inner_product
        >>>
        >>> array1 = np.array([1, 2, 3])
        >>> array2 = np.array([4, 5, 6])
        >>> inner_product(array1, array2)
        32

        If the arrays contain more than one sample

        >>> array1 = np.array([[1, 2, 3], [2, 3, 4]])
        >>> array2 = np.array([[4, 5, 6], [1, 1, 1]])
        >>> inner_product(array1, array2)
        array([32, 9])

        The inner product of the :math:'f(x) = x` and the constant
        :math:`y=1` defined over the interval [0,1] is the area of the
        triangle delimited by the the lines y = 0, x = 1 and y = x; 0.5.

        >>> import skfda
        >>>
        >>> x = np.linspace(0,1,1000)
        >>>
        >>> fd1 = skfda.FDataGrid(x,x)
        >>> fd2 = skfda.FDataGrid(np.ones(len(x)),x)
        >>> inner_product(fd1, fd2)
        array([ 0.5])

        If the FDataGrid object contains more than one sample

        >>> fd1 = skfda.FDataGrid([x, np.ones(len(x))], x)
        >>> fd2 = skfda.FDataGrid([np.ones(len(x)), x] ,x)
        >>> inner_product(fd1, fd2).round(2)
        array([ 0.5, 0.5])

        If one argument contains only one sample it is
        broadcasted.

        >>> fd1 = skfda.FDataGrid([x, np.ones(len(x))], x)
        >>> fd2 = skfda.FDataGrid([np.ones(len(x))] ,x)
        >>> inner_product(fd1, fd2).round(2)
        array([ 0.5, 1. ])

        It also work with basis objects

        >>> basis = skfda.representation.basis.Monomial(n_basis=3)
        >>>
        >>> fd1 = skfda.FDataBasis(basis, [0, 1, 0])
        >>> fd2 = skfda.FDataBasis(basis, [1, 0, 0])
        >>> inner_product(fd1, fd2)
        array([ 0.5])

        >>> basis = skfda.representation.basis.Monomial(n_basis=3)
        >>>
        >>> fd1 = skfda.FDataBasis(basis, [[0, 1, 0], [0, 0, 1]])
        >>> fd2 = skfda.FDataBasis(basis, [1, 0, 0])
        >>> inner_product(fd1, fd2)
        array([ 0.5       , 0.33333333])

        >>> basis = skfda.representation.basis.Monomial(n_basis=3)
        >>>
        >>> fd1 = skfda.FDataBasis(basis, [[0, 1, 0], [0, 0, 1]])
        >>> fd2 = skfda.FDataBasis(basis, [[1, 0, 0], [0, 1, 0]])
        >>> inner_product(fd1, fd2)
        array([ 0.5 , 0.25])

    """

    if callable(arg1):
        return _inner_product_integrate(arg1, arg2, matrix=matrix)
    else:
        return (np.einsum('n...,m...->nm...', arg1, arg2).sum(axis=-1)
                if matrix else (arg1 * arg2).sum(axis=-1))


@inner_product.register
def inner_product_fdatagrid(arg1: FDataGrid, arg2: FDataGrid, *, matrix=False):

    if not np.array_equal(arg1.grid_points,
                          arg2.grid_points):
        raise ValueError("Sample points for both objects must be equal")

    d1 = arg1.data_matrix
    d2 = arg2.data_matrix

    einsum_broadcast_list = (np.arange(d1.ndim - 1) + 2).tolist()

    if matrix:

        d1 = np.copy(d1)

        # Perform quadrature inside the einsum
        for i, s in enumerate(arg1.grid_points[::-1]):
            identity = np.eye(len(s))
            weights = scipy.integrate.simps(identity, x=s)
            index = (slice(None),) + (np.newaxis,) * (i + 1)
            d1 *= weights[index]

        return np.einsum(d1, [0] + einsum_broadcast_list,
                         d2, [1] + einsum_broadcast_list,
                         [0, 1])

    else:
        integrand = d1 * d2

        for s in arg1.grid_points[::-1]:
            integrand = scipy.integrate.simps(integrand,
                                              x=s,
                                              axis=-2)

        return np.sum(integrand, axis=-1)


@inner_product.register(FDataBasis, FDataBasis)
@inner_product.register(FDataBasis, Basis)
@inner_product.register(Basis, FDataBasis)
@inner_product.register(Basis, Basis)
def inner_product_fdatabasis(arg1: Union[FDataBasis, Basis],
                             arg2: Union[FDataBasis, Basis],
                             *,
                             matrix=False,
                             inner_product_matrix=None,
                             force_numerical=False):

    if not _same_domain(arg1, arg2):
        raise ValueError("Both Objects should have the same domain_range")

    if isinstance(arg1, Basis):
        arg1 = arg1.to_basis()

    if isinstance(arg2, Basis):
        arg2 = arg2.to_basis()

    # Now several cases where computing the matrix is preferrable
    #
    # First, if force_numerical is True, the matrix is NOT used
    # Otherwise, if the matrix is given, it is used
    # Two other cases follow

    # The basis is the same: most basis can optimize this case,
    # and also the Gram matrix is cached the first time, so computing
    # it is usually worthwhile
    same_basis = arg1.basis == arg2.basis

    # The number of operations is less using the matrix
    n_ops_best_with_matrix = max(
        arg1.n_samples, arg2.n_samples) > arg1.n_basis * arg2.n_basis

    if not force_numerical and (
            inner_product_matrix is not None
            or same_basis
            or n_ops_best_with_matrix):

        if inner_product_matrix is None:
            inner_product_matrix = arg1.basis.inner_product_matrix(arg2.basis)

        coef1 = arg1.coefficients
        coef2 = arg2.coefficients

        if matrix:
            return np.einsum('nb,bc,mc->nm',
                             coef1, inner_product_matrix, coef2)
        else:
            return (coef1 @
                    inner_product_matrix *
                    coef2).sum(axis=-1)

    else:
        return _inner_product_integrate(arg1, arg2, matrix=matrix)


def _inner_product_integrate(arg1, arg2, *, matrix=False):

    if not np.array_equal(arg1.domain_range,
                          arg2.domain_range):
        raise ValueError("Domain range for both objects must be equal")

    def integrand(*args):
        f1 = arg1([*args])[:, 0, :]
        f2 = arg2([*args])[:, 0, :]

        if matrix:
            ret = np.einsum('n...,m...->nm...', f1, f2)
            ret = ret.reshape((-1,) + ret.shape[2:])
            return ret
        else:
            return f1 * f2

    integral = nquad_vec(
        integrand,
        arg1.domain_range)

    summation = np.sum(integral, axis=-1)

    if matrix:
        summation = summation.reshape((len(arg1), len(arg2)))

    return summation


def inner_product_matrix(arg1, arg2=None, **kwargs):
    """
    Returns the inner product matrix between is arguments.

    If arg2 is ``None`` returns the Gram matrix.

    Args:

        arg1: First sample.
        arg2: Second sample.

    """

    if isinstance(arg1, Basis):
        arg1 = arg1.to_basis()
    if isinstance(arg2, Basis):
        arg2 = arg2.to_basis()

    if arg2 is None:
        arg2 = arg1

    return inner_product(arg1, arg2, matrix=True, **kwargs)
