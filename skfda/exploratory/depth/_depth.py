"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import itertools
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


class BandDepth(FunctionalDepth):
    """
    Functional depth as the integral of a multivariate depth.

    """

    def fit(self, X, y=None):

        if X.dim_codomain != 1:
            raise NotImplementedError("Band depth not implemented for vector "
                                      "valued functions")

        self._distribution = X
        return self

    def predict(self, X, *, pointwise=False):

        num_in = 0
        n_total = 0

        for f1, f2 in itertools.combinations(self._distribution, 2):
            between_range_1 = (f1.data_matrix <= X.data_matrix) & (
                X.data_matrix <= f2.data_matrix)

            between_range_2 = (f2.data_matrix <= X.data_matrix) & (
                X.data_matrix <= f1.data_matrix)

            between_range = between_range_1 | between_range_2

            num_in += np.all(between_range,
                             axis=tuple(range(1, X.data_matrix.ndim)))
            n_total += 1

        return num_in / n_total


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
        return BandDepth().fit(fdatagrid).predict(
            fdatagrid, pointwise=pointwise)


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
