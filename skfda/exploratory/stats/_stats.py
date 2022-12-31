"""Functional data descriptive statistics."""
from __future__ import annotations

from builtins import isinstance
from typing import TypeVar, Union

import numpy as np
from scipy import integrate
from scipy.stats import rankdata

from ...misc.metrics._lp_distances import l2_distance
from ...representation import FData, FDataGrid
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat
from ..depth import Depth, ModifiedBandDepth

F = TypeVar('F', bound=FData)
T = TypeVar('T', bound=Union[NDArrayFloat, FData])


def mean(
    X: F,
    weights: NDArrayFloat | None = None,
) -> F:
    """
    Compute the mean of all the samples in a FData object.

    Args:
        X: Object containing all the samples whose mean is wanted.
        weights: Sample weight. By default, uniform weight are used.

    Returns:
        Mean of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    if weights is None:
        return X.mean()

    weight = (1 / np.sum(weights)) * weights
    return (X * weight).sum()


def var(X: FData) -> FDataGrid:
    """
    Compute the variance of a set of samples in a FData object.

    Args:
        X: Object containing all the set of samples whose variance is desired.

    Returns:
        Variance of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    return X.var()  # type: ignore[no-any-return]


def gmean(X: FDataGrid) -> FDataGrid:
    """
    Compute the geometric mean of all the samples in a FDataGrid object.

    Args:
        X: Object containing all the samples whose geometric mean is wanted.

    Returns:
        Geometric mean of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    return X.gmean()


def cov(X: FData) -> FDataGrid:
    """
    Compute the covariance.

    Calculates the covariance matrix representing the covariance of the
    functional samples at the observation points.

    Args:
        X: Object containing different samples of a functional variable.

    Returns:
        Covariance of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    return X.cov()  # type: ignore[no-any-return]


def modified_epigraph_index(X: FDataGrid) -> NDArrayFloat:
    """
    Calculate the Modified Epigraph Index of a FDataGrid.

    The MEI represents the mean time a curve stays below other curve.
    In this case we will calculate the MEI for each curve in relation
    with all the other curves of our dataset.

    """
    interval_len = (
        X.domain_range[0][1]
        - X.domain_range[0][0]
    )

    # Array containing at each point the number of curves
    # are above it.
    num_functions_above: NDArrayFloat = rankdata(
        -X.data_matrix,
        method='max',
        axis=0,
    ) - 1

    integrand = num_functions_above

    for d, s in zip(X.domain_range, X.grid_points):
        integrand = integrate.simps(
            integrand,
            x=s,
            axis=1,
        )
        interval_len = d[1] - d[0]
        integrand /= interval_len

    integrand /= X.n_samples

    return integrand.flatten()


def depth_based_median(
    X: T,
    depth_method: Depth[T] | None = None,
) -> T:
    """
    Compute the median based on a depth measure.

    The depth based median is the deepest curve given a certain
    depth measure.

    Args:
        X: Object containing different samples of a
            functional variable.
        depth_method: Depth method used to order the data. Defaults to
            :func:`modified band
            depth <skfda.exploratory.depth.ModifiedBandDepth>`.

    Returns:
        Object containing the computed depth_based median.

    See also:
        :func:`geometric_median`

    """
    depth_method_used: Depth[T]

    if depth_method is None:
        assert isinstance(X, FDataGrid)
        depth_method_used = ModifiedBandDepth()
    else:
        depth_method_used = depth_method

    depth = depth_method_used(X)
    indices_descending_depth = (-depth).argsort(axis=0)

    # The median is the deepest curve
    return X[indices_descending_depth[0]]


def _weighted_average(X: T, weights: NDArrayFloat) -> T:

    if isinstance(X, FData):
        return (X * weights).sum()

    return (X.T * weights).T.sum(axis=0)  # type: ignore[no-any-return]


def geometric_median(
    X: T,
    *,
    tol: float = 1.e-8,
    metric: Metric[T] = l2_distance,
) -> T:
    r"""
    Compute the geometric median.

    The sample geometric median is the point that minimizes the :math:`L_1`
    norm of the vector of distances to all observations:

    .. math::

        \underset{y \in L(\mathcal{T})}{\arg \min}
        \sum_{i=1}^N \left \| x_i-y \right \|

    The geometric median in the functional case is also described in
    :footcite:`gervini_2008_estimation`.
    Instead of the proposed algorithm, however, the current implementation
    uses the corrected Weiszfeld algorithm to compute the median.

    Args:
        X: Object containing different samples of a
            functional variable.
        tol: tolerance used to check convergence.
        metric: metric used to compute the vector of distances. By
            default is the :math:`L_2` distance.

    Returns:
        Object containing the computed geometric median.

    Example:

        >>> from skfda import FDataGrid
        >>> data_matrix = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> X = FDataGrid(data_matrix)
        >>> median = geometric_median(X)
        >>> median.data_matrix[0, ..., 0]
        array([ 1. ,  1. ,  3. ,  0.5])

    See also:
        :func:`depth_based_median`

    References:
        .. footbibliography::

    """
    weights = np.full(len(X), 1 / len(X))
    median = _weighted_average(X, weights)
    distances = metric(X, median)

    while True:
        zero_distances = (distances == 0)
        n_zeros = np.sum(zero_distances)
        weights_new = (
            (1 / distances) / np.sum(1 / distances) if n_zeros == 0
            else (1 / n_zeros) * zero_distances
        )

        median_new = _weighted_average(X, weights_new)

        if l2_distance(median_new, median) < tol:
            return median_new

        distances = metric(X, median_new)

        weights, median = (weights_new, median_new)


def trim_mean(
    X: F,
    proportiontocut: float,
    *,
    depth_method: Depth[F] | None = None,
) -> FDataGrid:
    """Compute the trimmed means based on a depth measure.

    The trimmed means consists in computing the mean function without a
    percentage of least deep curves. That is, we first remove the least deep
    curves and then we compute the mean as usual.

    Note that in scipy the leftmost and rightmost proportiontocut data are
    removed. In this case, as we order the data by the depth, we only remove
    those that have the least depth values.

    Args:
        X: Object containing different samples of a
            functional variable.
        proportiontocut: Indicates the percentage of functions to
            remove. It is not easy to determine as it varies from dataset to
            dataset.
        depth_method: Method used to order the data. Defaults to
            :func:`modified band depth
            <skfda.exploratory.depth.ModifiedBandDepth>`.

    Returns:
        Object containing the computed trimmed mean.

    """
    if depth_method is None:
        depth_method = ModifiedBandDepth()

    n_samples_to_keep = (len(X) - int(len(X) * proportiontocut))

    # compute the depth of each curve and store the indexes in descending order
    depth = depth_method(X)
    indices_descending_depth = (-depth).argsort(axis=0)

    trimmed_curves = X[indices_descending_depth[:n_samples_to_keep]]

    return trimmed_curves.mean()
