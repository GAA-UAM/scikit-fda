"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import abc
from functools import reduce
import math

import scipy.integrate
from scipy.stats import rankdata

import numpy as np

from . import multivariate


__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


class FunctionalDepth(multivariate.Depth):
    """
    Abstract class representing a functional depth function.

    Usually it will accept a distribution in the initializer.

    """
    pass


def _cumulative_distribution(column):
    """Calculates the cumulative distribution function of the values passed to
    the function and evaluates it at each point.

    Args:
        column (numpy.darray): Array containing the values over which the
            distribution function is calculated.

    Returns:
        numpy.darray: Array containing the evaluation at each point of the
            distribution function.

    Examples:
        >>> _cumulative_distribution(np.array([1, 4, 5, 1, 2, 2, 4, 1, 1, 3]))
        array([ 0.4,  0.9,  1. ,  0.4,  0.6,  0.6,  0.9,  0.4,  0.4,  0.7])

    """
    if len(column.shape) != 1:
        raise ValueError("Only supported 1 dimensional arrays.")
    _, indexes, counts = np.unique(column, return_inverse=True,
                                   return_counts=True)
    count_cumulative = np.cumsum(counts) / len(column)
    return count_cumulative[indexes].reshape(column.shape)


class IntegratedDepth(FunctionalDepth):
    """
    Functional depth as the integral of a multivariate depth.

    """

    def __init__(self, *,
                 multivariate_depth=multivariate._UnivariateFraimanMuniz()):
        self.multivariate_depth = multivariate_depth

    def fit(self, X, y=None):

        self._domain_range = X.domain_range
        self._grid_points = X.grid_points
        self.multivariate_depth.fit(X.data_matrix)
        return self

    def predict(self, X, *, pointwise=False):

        pointwise_depth = self.multivariate_depth.predict(X.data_matrix)

        if pointwise:
            return pointwise_depth
        else:

            interval_len = (self._domain_range[0][1]
                            - self._domain_range[0][0])

            integrand = pointwise_depth

            for d, s in zip(X.domain_range, X.grid_points):
                integrand = scipy.integrate.simps(integrand,
                                                  x=s,
                                                  axis=1)
                interval_len = d[1] - d[0]
                integrand /= interval_len

            return integrand

    @property
    def max(self):
        return self.multivariate_depth.max

    @property
    def min(self):
        return self.multivariate_depth.min


class ModifiedBandDepth(IntegratedDepth):

    def __init__(self):
        super().__init__(multivariate_depth=multivariate.SimplicialDepth())


def outlyingness_to_depth(outlyingness, *, supreme=None):
    r"""Convert outlyingness function to depth function.

    An outlyingness function :math:`O(x)` can be converted to a depth
    function as

    .. math::
        D(x) = \frac{1}{1 + O(x)}

    if :math:`O(x)` is unbounded or as

    .. math::
        D(x) = 1 - \frac{O(x)}{\sup O(x)}

    if :math:`O(x)` is bounded ([Se06]_).

    Args:
        outlyingness (Callable): Outlyingness function.
        supreme (float, optional): Supreme value of the outlyingness function.

    Returns:
        Callable: The corresponding depth function.

    References:
        .. [Se06] Serfling, R. (2006). Depth functions in nonparametric
           multivariate inference. DIMACS Series in Discrete Mathematics and
           Theoretical Computer Science, 72, 1.
    """

    if supreme is None or math.isinf(supreme):
        def depth(*args, **kwargs):
            return 1 / (1 + outlyingness(*args, **kwargs))
    else:
        def depth(*args, **kwargs):
            return 1 - outlyingness(*args, **kwargs) / supreme

    return depth


def _rank_samples(fdatagrid):
    """Ranks the he samples in the FDataGrid at each point of discretisation.

    Args:
        fdatagrid (FDataGrid): Object whose samples are ranked.

    Returns:
        numpy.darray: Array containing the ranks of the sample points.

    Examples:
        Univariate setting:

        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> _rank_samples(fd)
        array([[ 4.,  4.,  4.,  4.,  4.,  4.],
               [ 3.,  3.,  3.,  3.,  3.,  3.],
               [ 1.,  1.,  2.,  2.,  2.,  2.],
               [ 2.,  2.,  2.,  1.,  1.,  1.]])

        Several input dimensions:

        >>> data_matrix = [[[[1], [0.7], [1]],
        ...                 [[4], [0.4], [5]]],
        ...                [[[2], [0.5], [2]],
        ...                 [[3], [0.6], [3]]]]
        >>> grid_points = [[2, 4], [3, 6, 8]]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> _rank_samples(fd)
        array([[[ 1.,  2.,  1.],
                [ 2.,  1.,  2.]],
               [[ 2.,  1.,  2.],
                [ 1.,  2.,  1.]]])



    """
    if fdatagrid.dim_codomain > 1:
        raise ValueError("Currently multivariate data is not allowed")

    ranks = np.zeros(fdatagrid.data_matrix.shape[:-1])

    for index, _ in np.ndenumerate(ranks[0]):
        ranks[(slice(None),) + index] = rankdata(
            fdatagrid.data_matrix[(slice(None),) + index + (0,)], method='max')
    return ranks


def band_depth(fdatagrid, *, pointwise=False):
    """Implementation of Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of the
    bands determined by two sample curves containing the whole graph of the
    first one. In the case the fdatagrid domain dimension is 2, instead of
    curves, surfaces determine the bands. In larger dimensions, the hyperplanes
    determine the bands.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the band depth is
            going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is
            returned instead. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the band depth of the samples,
            or the band depth of the samples at each point of discretization
            if pointwise equals to True.

    Examples:

        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> band_depth(fd)
        array([ 0.5       ,  0.83333333,  0.5       ,  0.5       ])

    """
    if pointwise:
        return modified_band_depth(fdatagrid, pointwise)
    else:
        n = fdatagrid.n_samples
        nchoose2 = n * (n - 1) / 2

        ranks = _rank_samples(fdatagrid)
        axis = tuple(range(1, fdatagrid.dim_domain + 1))
        n_samples_above = fdatagrid.n_samples - np.amax(ranks, axis=axis)
        n_samples_below = np.amin(ranks, axis=axis) - 1
        depth = ((n_samples_below * n_samples_above + fdatagrid.n_samples - 1)
                 / nchoose2)

        return depth


def modified_band_depth(fdatagrid, *, pointwise=False):
    """Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time
    its graph is contained in the bands determined by two sample curves.
    In the case the fdatagrid domain dimension is 2, instead of curves,
    surfaces determine the bands. In larger dimensions, the hyperplanes
    determine the bands.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the modified band
            depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is
            returned instead. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the modified band depth of the
            samples, or the modified band depth of the samples at each point
            of discretization if pointwise equals to True.

    Examples:

        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> depth = modified_band_depth(fd)
        >>> depth.round(2)
        array([ 0.5 ,  0.83,  0.73,  0.67])
        >>> pointwise = modified_band_depth(fd, pointwise = True)
        >>> pointwise.round(2)
        array([[ 0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
               [ 0.83,  0.83,  0.83,  0.83,  0.83,  0.83],
               [ 0.5 ,  0.5 ,  0.83,  0.83,  0.83,  0.83],
               [ 0.83,  0.83,  0.83,  0.5 ,  0.5 ,  0.5 ]])

    """
    return ModifiedBandDepth().fit(fdatagrid).predict(
        fdatagrid, pointwise=pointwise)


def fraiman_muniz_depth(fdatagrid, *, pointwise=False):
    r"""Implementation of Fraiman and Muniz (FM) Depth for functional data.

    Each column is considered as the samples of an aleatory variable.
    The univariate depth of each of the samples of each column is calculated
    as follows:

    .. math::
        D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert

    Where :math:`F` stands for the marginal univariate distribution function of
    each column.

    The depth of a sample is the result of integrating the previously computed
    depth for each of its points and normalizing dividing by the length of
    the interval.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the FM depth is going
            to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is
             returned instead. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the Fraiman-Muniz depth of the
            samples, or the Fraiman-Muniz of the samples at each point
            of discretization if pointwise equals to True.

    Examples:
        Currently, this depth function can only be used
        for univariate functional data:

        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> fraiman_muniz_depth(fd)
        array([ 0.5  ,  0.75 ,  0.925,  0.875])

        You can use ``pointwise`` to obtain the pointwise depth,
        before the integral is applied.

        >>> pointwise = fraiman_muniz_depth(fd, pointwise = True)
        >>> pointwise
        array([[ 0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
               [ 0.75,  0.75,  0.75,  0.75,  0.75,  0.75],
               [ 0.75,  0.75,  1.  ,  1.  ,  1.  ,  1.  ],
               [ 1.  ,  1.  ,  1.  ,  0.75,  0.75,  0.75]])


    """
    return IntegratedDepth().fit(fdatagrid).predict(
        fdatagrid, pointwise=pointwise)
