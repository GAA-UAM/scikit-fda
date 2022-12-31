"""Maxima Hunting dimensionality reduction and related methods."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.signal
import sklearn.utils
from sklearn.base import clone

from dcor import u_distance_correlation_sqr

from ...._utils._sklearn_adapter import (
    BaseEstimator,
    InductiveTransformerMixin,
)
from ....representation import FDataGrid
from ....typing._numpy import NDArrayFloat, NDArrayInt, NDArrayReal
from ._base import _compute_dependence, _DependenceMeasure

_LocalMaximaSelector = Callable[[FDataGrid], NDArrayInt]


def _select_relative_maxima(X: FDataGrid, *, order: int = 1) -> NDArrayInt:

    X_array = X.data_matrix[0, ..., 0]

    indexes = scipy.signal.argrelextrema(
        X_array,
        comparator=np.greater_equal,
        order=order,
    )[0]

    # Discard flat
    maxima = X_array[indexes]

    left_points = np.take(X_array, indexes - 1, mode='clip')
    right_points = np.take(X_array, indexes + 1, mode='clip')

    is_not_flat = (maxima > left_points) | (maxima > right_points)

    return indexes[is_not_flat]  # type: ignore [no-any-return]


def select_local_maxima(X: FDataGrid, *, order: int = 1) -> NDArrayInt:
    r"""
    Compute local maxima of an array.

    Points near the boundary are considered maxima looking only at one side.

    For flat regions only the boundary points of the flat region could be
    considered maxima.

    Parameters:
        X: Where to compute the local maxima.
        order: How many points on each side to look, to check if
            a point is a maximum in that interval.

    Returns:
        Indexes of the local maxima.

    Examples:
        >>> from skfda import FDataGrid
        >>> from skfda.preprocessing.dim_reduction.variable_selection.\
        ...     maxima_hunting import select_local_maxima
        >>> import numpy as np

        >>> x = FDataGrid(np.array([2, 1, 1, 1, 2, 3, 3, 3, 2, 3, 4, 3, 2]))
        >>> select_local_maxima(x).astype(np.int_)
        array([ 0,  5,  7, 10])

        The ``order`` parameter can be used to check a larger interval to see
        if a point is still a maxima, effectively eliminating small local
        maxima.

        >>> x = FDataGrid(np.array([2, 1, 1, 1, 2, 3, 3, 3, 2, 3, 4, 3, 2]))
        >>> select_local_maxima(x, order=3).astype(np.int_)
        array([ 0,  5, 10])

    """
    return _select_relative_maxima(X, order=order)


class RelativeLocalMaximaSelector(BaseEstimator):

    def __init__(
        self,
        smoothing_parameter: int = 1,
        max_points: int | None = None,
    ):
        self.smoothing_parameter = smoothing_parameter
        self.max_points = max_points

    def __call__(self, X: FDataGrid) -> NDArrayInt:
        indexes = _select_relative_maxima(
            X,
            order=self.smoothing_parameter,
        )

        if self.max_points is not None:
            values = X.data_matrix[:, indexes]
            partition_indexes = np.argpartition(
                values,
                -self.max_points,
                axis=None,
            )
            indexes = indexes[np.sort(partition_indexes[-self.max_points:])]

        return indexes


class MaximaHunting(
    BaseEstimator,
    InductiveTransformerMixin[
        FDataGrid,
        NDArrayFloat,
        NDArrayReal,
    ],
):
    r"""
    Maxima Hunting variable selection.

    This is a filter variable selection method for problems with a target
    variable. It evaluates a dependence measure between each point of the
    function and the target variable, and keeps those points in which this
    dependence is a local maximum.

    Selecting the local maxima serves two purposes. First, it ensures that
    the points that are relevant in isolation are selected, as they must
    maximice their dependence with the target variable. Second, the points
    that are relevant only because they are near a relevant point (and are
    thus highly correlated with it) are NOT selected, as only local maxima
    are selected, minimizing the redundancy of the selected variables.

    For a longer explanation about the method, and comparison with other
    functional variable selection methods, we refer the reader to the
    original article :footcite:`berrendero+cuevas+torrecilla_2016_hunting`.

    Parameters:
        dependence_measure (callable): Dependence measure to use. By default,
            it uses the bias corrected squared distance correlation.
        local_maxima_selector (callable): Function to detect local maxima. The
            default is :func:`select_local_maxima` with ``order`` parameter
            equal to one. The original article used a similar function testing
            different values of ``order``.

    Examples:
        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.preprocessing.dim_reduction.variable_selection.\
        ...     maxima_hunting import RelativeLocalMaximaSelector
        >>> from skfda.datasets import make_gaussian_process
        >>> import skfda
        >>> import numpy as np

        We create trajectories from two classes, one with zero mean and the
        other with a peak-like mean. Both have Brownian covariance.

        >>> n_samples = 10000
        >>> n_features = 100
        >>>
        >>> def mean_1(t):
        ...     return (np.abs(t - 0.25)
        ...             - 2 * np.abs(t - 0.5)
        ...             + np.abs(t - 0.75))
        >>>
        >>> X_0 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     random_state=0,
        ... )
        >>> X_1 = make_gaussian_process(
        ...     n_samples=n_samples // 2,
        ...     n_features=n_features,
        ...     mean=mean_1,
        ...     random_state=1,
        ... )
        >>> X = skfda.concatenate((X_0, X_1))
        >>>
        >>> y = np.zeros(n_samples)
        >>> y [n_samples // 2:] = 1

        Select the relevant points to distinguish the two classes

        >>> local_maxima_selector = RelativeLocalMaximaSelector(
        ...     smoothing_parameter=10,
        ... )
        >>> mh = variable_selection.MaximaHunting(
        ...            local_maxima_selector=local_maxima_selector,
        ... )
        >>> _ = mh.fit(X, y)
        >>> point_mask = mh.get_support()
        >>> points = X.grid_points[0][point_mask]
        >>> np.allclose(points, [0.5], rtol=0.1)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = mh.transform(X)
        >>> len(X.grid_points[0])
        100
        >>> X_dimred.shape
        (10000, 1)

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        dependence_measure: _DependenceMeasure[
            NDArrayFloat,
            NDArrayFloat,
        ] = u_distance_correlation_sqr,
        local_maxima_selector: _LocalMaximaSelector | None = None,
    ) -> None:
        self.dependence_measure = dependence_measure
        self.local_maxima_selector = local_maxima_selector

    def fit(  # type: ignore[override] # noqa: D102
        self,
        X: FDataGrid,
        y: NDArrayReal,
    ) -> MaximaHunting:

        self._maxima_selector: Callable[[NDArrayFloat], NDArrayInt] = (
            RelativeLocalMaximaSelector()
            if self.local_maxima_selector is None
            else clone(self.local_maxima_selector, safe=False)
        )

        self.features_shape_ = X.data_matrix.shape[1:]
        self.dependence_ = _compute_dependence(
            X,
            y.astype(np.float_),
            dependence_measure=self.dependence_measure,
        )

        self.indexes_ = self._maxima_selector(
            self.dependence_,
        )

        sorting_indexes = np.argsort(
            self.dependence_.data_matrix[0, self.indexes_, 0])[::-1]
        self.sorted_indexes_ = self.indexes_[sorting_indexes]

        return self

    def get_support(self, indices: bool = False) -> NDArrayInt:  # noqa: D102
        if indices:
            return self.indexes_

        mask = np.zeros(self.features_shape_[:-1], dtype=bool)
        mask[self.indexes_] = True
        return mask

    def transform(  # noqa: D102
        self,
        X: FDataGrid,
        y: NDArrayInt | NDArrayFloat | None = None,
    ) -> NDArrayFloat:

        sklearn.utils.validation.check_is_fitted(self)

        if X.data_matrix.shape[1:] != self.features_shape_:
            raise ValueError(
                "The trajectories have a different number of "
                "points than the ones fitted",
            )

        return X.data_matrix[:, self.sorted_indexes_].reshape(X.n_samples, -1)
