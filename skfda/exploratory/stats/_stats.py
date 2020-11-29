"""Functional data descriptive statistics.
"""
from builtins import isinstance

from typing import Callable, TypeVar, Union

import numpy as np

from ...misc.metrics import lp_norm, lp_distance
from ...representation import FData
from ..depth import ModifiedBandDepth


def mean(fdata):
    """Compute the mean of all the samples in a FData object.

    Computes the mean of all the samples in a FDataGrid or FDataBasis object.

    Args:
        fdata (FDataGrid or FDataBasis): Object containing all the samples
            whose mean is wanted.


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


def depth_based_median(fdatagrid, depth_method=ModifiedBandDepth()):
    """Compute the median based on a depth measure.

    The depth based median is the deepest curve given a certain
    depth measure

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.
        depth_method (:ref:`depth measure <depth-measures>`, optional):
                Method used to order the data. Defaults to :func:`modified
                band depth <skfda.exploratory.depth.ModifiedBandDepth>`.

    Returns:
        FDataGrid: object containing the computed depth_based median.

    See also:
        geometric_median

    """
    depth = depth_method(fdatagrid)
    indices_descending_depth = (-depth).argsort(axis=0)

    # The median is the deepest curve
    return fdatagrid[indices_descending_depth[0]]


T = TypeVar('T', bound=Union[np.array, FData])


def geometric_median(X: T, tol: float=1.e-8,
                     metric: Callable = lp_distance) -> T:
    r"""Compute the geometric median.

    The sample geometric median is the point that minimizes the :math:`L_1`
    norm of the vector of Euclidean distances to all observations:

    .. math::

        \underset{y \in L(\mathcal{T})}{\arg \min}
        \sum_{i=1}^m \left \| x_i-y \right \|_2

    It uses the corrected Weiszfeld algorithm to compute the median,
    precalculating the inner product matrix in order to compute the
    distances.

    Args:
        X: Object containing different samples of a
            functional variable.
        tol: tolerance used to check convergence.

    Returns:
        FData: object containing the computed geometric median.

    Example:

        >>> from skfda import FDataGrid
        >>> data_matrix = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> X = FDataGrid(data_matrix)
        >>> median = geometric_median(X)
        >>> median.data_matrix[0, ..., 0]
        array([ 1. ,  1. ,  3. ,  0.5])

    See also:
        depth_based_median

    References:
        Gervini, D. (2008). Robust functional estimation using the median and
        spherical principal components. Biometrika, 95(3), 587–600.
        https://doi.org/10.1093/biomet/asn031

    """

    def weighted_average(X, weights):
        if isinstance(X, FData):
            return (X * weights).sum()
        else:
            # To support also multivariate data
            return (X.T * weights).T.sum(axis=0)

    weights = np.full(len(X), 1 / len(X))
    median = weighted_average(X, weights)
    distances = metric(X, median)

    while True:
        zero_distances = (distances == 0)
        n_zeros = np.sum(zero_distances)
        weights_new = ((1 / distances) / np.sum(1 / distances) if n_zeros == 0
                       else (1 / n_zeros) * zero_distances)

        median_new = weighted_average(X, weights_new)

        if lp_norm(median_new - median) < tol:
            return median_new

        distances = metric(X, median_new)

        weights, median = (weights_new, median_new)


def trim_mean(fdatagrid,
              proportiontocut,
              depth_method=ModifiedBandDepth()):
    """Compute the trimmed means based on a depth measure.

    The trimmed means consists in computing the mean function without a
    percentage of least deep curves. That is, we first remove the least deep
    curves and then we compute the mean as usual.

    Note that in scipy the leftmost and rightmost proportiontocut data are
    removed. In this case, as we order the data by the depth, we only remove
    those that have the least depth values.

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.
        proportiontocut (float): indicates the percentage of functions to
            remove. It is not easy to determine as it varies from dataset to
            dataset.
        depth_method (:ref:`depth measure <depth-measures>`, optional):
            Method used to order the data. Defaults to :func:`modified
            band depth <skfda.exploratory.depth.ModifiedBandDepth>`.

    Returns:
        FDataGrid: object containing the computed trimmed mean.

    """
    n_samples_to_keep = (fdatagrid.n_samples -
                         int(fdatagrid.n_samples * proportiontocut))

    # compute the depth of each curve and store the indexes in descending order
    depth = depth_method(fdatagrid)
    indices_descending_depth = (-depth).argsort(axis=0)

    trimmed_curves = fdatagrid[indices_descending_depth[:n_samples_to_keep]]

    return trimmed_curves.mean()
