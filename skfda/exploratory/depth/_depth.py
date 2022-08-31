"""
Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones).

"""
from __future__ import annotations

import itertools
from typing import TypeVar

import numpy as np
import scipy.integrate

from ..._utils._sklearn_adapter import BaseEstimator
from ...misc.metrics import l2_distance
from ...misc.metrics._utils import _fit_metric
from ...representation import FData, FDataGrid
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat
from .multivariate import Depth, SimplicialDepth, _UnivariateFraimanMuniz

T = TypeVar("T", bound=FData)


class IntegratedDepth(Depth[FDataGrid]):
    r"""
    Functional depth as the integral of a multivariate depth.

    Args:
        multivariate_depth (Depth): Multivariate depth to integrate.
            By default it is the one used by Fraiman and Muniz, that is,

            .. math::
                D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert

    Examples:
        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> depth = skfda.exploratory.depth.IntegratedDepth()
        >>> depth(fd)
        array([ 0.5  ,  0.75 ,  0.925,  0.875])

    References:
        Fraiman, R., & Muniz, G. (2001). Trimmed means for functional
        data. Test, 10(2), 419–440. https://doi.org/10.1007/BF02595706


    """

    def __init__(
        self,
        *,
        multivariate_depth: Depth[NDArrayFloat] | None = None,
    ) -> None:
        self.multivariate_depth = multivariate_depth

    def fit(  # noqa: D102
        self,
        X: FDataGrid,
        y: object = None,
    ) -> IntegratedDepth:

        self.multivariate_depth_: Depth[NDArrayFloat]

        if self.multivariate_depth is None:
            self.multivariate_depth_ = _UnivariateFraimanMuniz()
        else:
            self.multivariate_depth_ = self.multivariate_depth

        self._domain_range = X.domain_range
        self._grid_points = X.grid_points
        self.multivariate_depth_.fit(X.data_matrix)
        return self

    def transform(self, X: FDataGrid) -> NDArrayFloat:  # noqa: D102

        pointwise_depth = self.multivariate_depth_.transform(X.data_matrix)

        interval_len = (
            self._domain_range[0][1]
            - self._domain_range[0][0]
        )

        integrand = pointwise_depth

        for d, s in zip(X.domain_range, X.grid_points):
            integrand = scipy.integrate.simps(
                integrand,
                x=s,
                axis=1,
            )
            interval_len = d[1] - d[0]
            integrand /= interval_len

        return integrand

    @property
    def max(self) -> float:
        if self.multivariate_depth is None:
            return 1

        return self.multivariate_depth.max

    @property
    def min(self) -> float:
        if self.multivariate_depth is None:
            return 1 / 2

        return self.multivariate_depth.min


class ModifiedBandDepth(IntegratedDepth):
    """
    Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time
    its graph is contained in the bands determined by two sample curves.
    In the case the fdatagrid :term:`domain` dimension is 2, instead of curves,
    surfaces determine the bands. In larger dimensions, the hyperplanes
    determine the bands.

    Examples:
        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> depth = skfda.exploratory.depth.ModifiedBandDepth()
        >>> values = depth(fd)
        >>> values.round(2)
        array([ 0.5 ,  0.83,  0.73,  0.67])

    References:
        López-Pintado, S., & Romo, J. (2009). On the Concept of
        Depth for Functional Data. Journal of the American Statistical
        Association, 104(486), 718–734.
        https://doi.org/10.1198/jasa.2009.0108
    """

    def __init__(self) -> None:
        super().__init__(multivariate_depth=SimplicialDepth())


class BandDepth(Depth[FDataGrid]):
    """
    Implementation of Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of the
    bands determined by two sample curves containing the whole graph of the
    first one. In the case the fdatagrid :term:`domain` dimension is 2, instead
    of curves, surfaces determine the bands. In larger dimensions, the
    hyperplanes determine the bands.

    Examples:
        >>> import skfda
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> depth = skfda.exploratory.depth.BandDepth()
        >>> depth(fd)
        array([ 0.5       ,  0.83333333,  0.5       ,  0.5       ])

    References:
        López-Pintado, S., & Romo, J. (2009). On the Concept of
        Depth for Functional Data. Journal of the American Statistical
        Association, 104(486), 718–734.
        https://doi.org/10.1198/jasa.2009.0108

    """

    def fit(self, X: FDataGrid, y: object = None) -> BandDepth:  # noqa: D102

        if X.dim_codomain != 1:
            raise NotImplementedError(
                "Band depth not implemented for vector valued functions",
            )

        self._distribution = X
        return self

    def transform(self, X: FDataGrid) -> NDArrayFloat:  # noqa: D102

        num_in = np.zeros(shape=len(X), dtype=X.data_matrix.dtype)
        n_total = 0

        for f1, f2 in itertools.combinations(self._distribution, 2):
            between_range_1 = (
                (f1.data_matrix <= X.data_matrix)
                & (X.data_matrix <= f2.data_matrix)
            )

            between_range_2 = (
                (f2.data_matrix <= X.data_matrix)
                & (X.data_matrix <= f1.data_matrix)
            )

            between_range = between_range_1 | between_range_2

            num_in += np.all(
                between_range,
                axis=tuple(range(1, X.data_matrix.ndim)),
            )
            n_total += 1

        return num_in / n_total


class DistanceBasedDepth(Depth[FDataGrid], BaseEstimator):
    r"""
    Functional depth based on a metric.

    Parameters:
        metric:
            The metric to use as M in the following depth calculation

            .. math::
                D(x) = [1 + M(x, \mu)]^{-1}.

            as explained in :footcite:`serfling+zuo_2000_depth_function`.

    Examples:
        >>> import skfda
        >>> from skfda.exploratory.depth import DistanceBasedDepth
        >>> from skfda.misc.metrics import MahalanobisDistance
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> depth = DistanceBasedDepth(MahalanobisDistance(2))
        >>> depth(fd)
        array([ 0.41897777,  0.8058132 ,  0.31097392,  0.31723619])

    References:
        .. footbibliography::
    """

    def __init__(
        self,
        metric: Metric[T] = l2_distance,
    ) -> None:
        self.metric = metric

    def fit(  # noqa: D102
        self,
        X: T,
        y: object = None,
    ) -> DistanceBasedDepth:
        """Fit the model using X as training data.

        Args:
            X: FDataGrid with the training data or array matrix with shape
                (n_samples, n_samples) if metric='precomputed'.
            y: Ignored.

        Returns:
            self
        """
        _fit_metric(self.metric, X)

        self.mean_ = X.mean()

        return self

    def transform(self, X: T) -> NDArrayFloat:  # noqa: D102
        """Compute the depth of given observations.

        Args:
            X: FDataGrid with the observations to use in the calculation.

        Returns:
            Array with the depths.
        """
        return 1 / (1 + self.metric(X, self.mean_))
