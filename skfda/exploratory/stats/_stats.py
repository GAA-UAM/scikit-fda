"""Functional data descriptive statistics."""
from __future__ import annotations

import functools
from builtins import isinstance
from typing import Callable, TypeVar, Union

import numpy as np
from scipy.stats import rankdata

from skfda._utils.ndfunction import average_function_value

from ...misc.metrics._lp_distances import l2_distance
from ...representation import FData, FDataBasis, FDataGrid, FDataIrregular
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


def var(X: FData, correction: int = 0) -> FDataGrid:
    """
    Compute the variance of a set of samples in a FData object.

    Args:
        X: Object containing all the set of samples whose variance is desired.
        correction: degrees of freedom adjustment. The divisor used in the
            calculation is `N - correction`, where `N` represents the number of
            elements. Default: `0`.

    Returns:
        Variance of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    return X.var(correction=correction)  # type: ignore[no-any-return]


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


def cov(
    X: FData,
    correction: int = 0,
) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]:
    """
    Compute the covariance.

    Calculates the covariance matrix representing the covariance of the
    functional samples at the observation points.

    Args:
        X: Object containing different samples of a functional variable.
        correction: degrees of freedom adjustment. The divisor used in the
            calculation is `N - correction`, where `N` represents the number of
            elements. Default: `0`.


    Returns:
        Covariance of all the samples in the original object, as a
        callable.

    """
    return X.cov(correction=correction)


@functools.singledispatch
def std(X: F, correction: int = 0) -> F:
    r"""
    Compute the standard deviation of all the samples in a FData object.

    .. math::
        \text{std}_X(t) = \sqrt{\frac{1}{N-\text{correction}}
        \sum_{n=1}^{N}{\left(X_n(t) - \overline{X}(t)\right)^2}}

    Args:
        X: Object containing all the samples whose standard deviation is
            wanted.
        correction: degrees of freedom adjustment. The divisor used in the
            calculation is `N - correction`, where `N` represents the number of
            elements. Default: `0`.

    Returns:
        Standard deviation of all the samples in the original object, as a
        :term:`functional data object` with just one sample.

    """
    raise NotImplementedError("Not implemented for this type")


@std.register
def std_fdatagrid(X: FDataGrid, correction: int = 0) -> FDataGrid:
    """Compute the standard deviation of a FDataGrid."""
    return X.copy(
        data_matrix=np.std(
            X.data_matrix, axis=0, ddof=correction,
        )[np.newaxis, ...],
        sample_names=(None,),
    )


@std.register
def std_fdatairregular(
    X: FDataIrregular, correction: int = 0,
) -> FDataIrregular:
    """Compute the standard deviation of a FDataIrregular."""
    common_points, common_values = X._get_common_points_and_values()
    std_values = np.std(
        common_values, axis=0, ddof=correction,
    )

    return FDataIrregular(
        start_indices=np.array([0]),
        points=common_points,
        values=std_values,
        sample_names=(None,),
    )


@std.register
def std_fdatabasis(X: FDataBasis, correction: int = 0) -> FDataBasis:
    """Compute the standard deviation of a FDataBasis."""
    from ..._utils import function_to_fdatabasis

    basis = X.basis
    coeff_cov_matrix = np.cov(
        X.coefficients, rowvar=False, ddof=correction,
    ).reshape((basis.n_basis, basis.n_basis))

    def std_function(t_points: NDArrayFloat) -> NDArrayFloat:  # noqa: WPS430
        basis_evaluation = basis(t_points).reshape((basis.n_basis, -1))
        std_values = np.sqrt(
            np.sum(
                basis_evaluation * (coeff_cov_matrix @ basis_evaluation),
                axis=0,
            ),
        )
        return np.reshape(std_values, (1, -1, X.dim_codomain))

    return function_to_fdatabasis(f=std_function, new_basis=X.basis)


def modified_epigraph_index(X: FDataGrid) -> NDArrayFloat:
    """
    Calculate the Modified Epigraph Index of a FDataGrid.

    The MEI represents the mean time a curve stays below other curve.
    In this case we will calculate the MEI for each curve in relation
    with all the other curves of our dataset.

    """
    # Functions containing at each point the number of curves
    # are above it.
    num_functions_above = X.copy(
        data_matrix=rankdata(
            -X.data_matrix,
            method='max',
            axis=0,
        ) - 1,
    )

    return (
        average_function_value(num_functions_above)
        / num_functions_above.n_samples
    ).ravel()


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
    :footcite:`gervini_2008_robust`.
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
