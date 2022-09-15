"""Depth and outlyingness ABCs and implementations for multivariate data."""

from __future__ import annotations

import abc
import math
from typing import TypeVar

import numpy as np
import scipy.stats
import sklearn
from scipy.special import comb
from typing_extensions import Literal

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...typing._numpy import NDArrayFloat, NDArrayInt

T = TypeVar("T", contravariant=True)
SelfType = TypeVar("SelfType")
_Side = Literal["left", "right"]
Input = TypeVar("Input", contravariant=True)


class _DepthOrOutlyingness(
    BaseEstimator,
    InductiveTransformerMixin[Input, NDArrayFloat, object],
):
    """Abstract class representing a depth or outlyingness function."""

    def fit(self: SelfType, X: Input, y: object = None) -> SelfType:
        """
        Learn the distribution from the observations.

        Args:
            X: Functional dataset from which the distribution of the data is
                inferred.
            y: Unused. Kept only for convention.

        Returns:
            Fitted estimator.

        """
        return self

    @abc.abstractmethod
    def transform(self, X: Input) -> NDArrayFloat:
        """
        Compute the depth or outlyingness inside the learned distribution.

        Args:
            X: Points whose depth is going to be evaluated.

        Returns:
            Depth of each observation.

        """
        pass

    def fit_transform(self, X: Input, y: object = None) -> NDArrayFloat:
        """
        Compute the depth or outlyingness of each observation.

        This computation is done with respect to the whole dataset.

        Args:
            X: Dataset.
            y: Unused. Kept only for convention.

        Returns:
            Depth of each observation.

        """
        return self.fit(X).transform(X)

    def __call__(
        self,
        X: Input,
        *,
        distribution: Input | None = None,
    ) -> NDArrayFloat:
        """
        Allow the depth or outlyingness to be used as a function.

        Args:
            X: Points whose depth is going to be evaluated.
            distribution: Functional dataset from which the distribution of
                the data is inferred. If ``None`` it is the same as ``X``.

        Returns:
            Depth of each observation.

        """
        copy: _DepthOrOutlyingness[Input] = sklearn.base.clone(self)

        if distribution is None:
            return copy.fit_transform(X)

        return copy.fit(distribution).transform(X)

    @property
    def max(self) -> float:
        """
        Maximum (or supremum if there is no maximum) of the possibly predicted
        values.

        """
        return 1

    @property
    def min(self) -> float:
        """
        Minimum (or infimum if there is no maximum) of the possibly predicted
        values.

        """
        return 0


class Depth(_DepthOrOutlyingness[T]):
    """Abstract class representing a depth function."""


class Outlyingness(_DepthOrOutlyingness[T]):
    """Abstract class representing an outlyingness function."""


def _searchsorted_one_dim(
    array: NDArrayFloat,
    values: NDArrayFloat,
    *,
    side: _Side = 'left',
) -> NDArrayInt:
    return np.searchsorted(array, values, side=side)


_searchsorted_vectorized = np.vectorize(
    _searchsorted_one_dim,
    signature='(n),(m),()->(m)',
    excluded='side',
)


def _searchsorted_ordered(
    array: NDArrayFloat,
    values: NDArrayFloat,
    *,
    side: _Side = 'left',
) -> NDArrayInt:
    return _searchsorted_vectorized(  # type: ignore[no-any-return]
        array,
        values,
        side=side,
    )


def _cumulative_distribution(column: NDArrayFloat) -> NDArrayFloat:
    """
    Calculate the cumulative distribution function at each point.

    Args:
        column: Array containing the values over which the
            distribution function is calculated.

    Returns:
        Array containing the evaluation at each point of the
        distribution function.

    Examples:
        >>> _cumulative_distribution(np.array([1, 4, 5, 1, 2, 2, 4, 1, 1, 3]))
        array([ 0.4,  0.9,  1. ,  0.4,  0.6,  0.6,  0.9,  0.4,  0.4,  0.7])

    """
    return _searchsorted_ordered(
        np.sort(column),
        column,
        side='right',
    ) / len(column)


class _UnivariateFraimanMuniz(Depth[NDArrayFloat]):
    r"""
    Univariate depth used to compute the Fraiman an Muniz depth.

    Each column is considered as the samples of an aleatory variable.
    The univariate depth of each of the samples of each column is calculated
    as follows:

    .. math::
        D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert

    Where :math:`F` stands for the marginal univariate distribution function of
    each column.

    """

    def fit(self: SelfType, X: NDArrayFloat, y: object = None) -> SelfType:
        self._sorted_values = np.sort(X, axis=0)
        return self

    def transform(self, X: NDArrayFloat) -> NDArrayFloat:
        cum_dist = _searchsorted_ordered(
            np.moveaxis(self._sorted_values, 0, -1),
            np.moveaxis(X, 0, -1),
            side='right',
        ).astype(X.dtype) / len(self._sorted_values)

        assert cum_dist.shape[-2] == 1
        ret = 0.5 - np.moveaxis(cum_dist, -1, 0)[..., 0]
        ret = - np.abs(ret)
        ret += 1

        return ret

    @property
    def min(self) -> float:
        return 1 / 2


class SimplicialDepth(Depth[NDArrayFloat]):
    r"""
    Simplicial depth.

    The simplicial depth of a point :math:`x` in :math:`\mathbb{R}^p` given a
    distribution :math:`F` is the probability that a random simplex with its
    :math:`p + 1` points sampled from :math:`F` contains :math:`x`.

    References:
        Liu, R. Y. (1990). On a Notion of Data Depth Based on Random
        Simplices. The Annals of Statistics, 18(1), 405–414.


    """

    def fit(  # noqa: D102
        self,
        X: NDArrayFloat,
        y: object = None,
    ) -> SimplicialDepth:
        self._dim = X.shape[-1]

        if self._dim == 1:
            self.sorted_values = np.sort(X, axis=0)
        else:
            raise NotImplementedError(
                "SimplicialDepth is currently only "
                "implemented for one-dimensional data.",
            )

        return self

    def transform(self, X: NDArrayFloat) -> NDArrayFloat:  # noqa: D102

        assert self._dim == X.shape[-1]

        if self._dim == 1:
            positions_left = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1),
            )

            positions_left = np.moveaxis(positions_left, -1, 0)[..., 0]

            positions_right = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1),
                side='right',
            )

            positions_right = np.moveaxis(positions_right, -1, 0)[..., 0]

            num_strictly_below = positions_left
            num_strictly_above = len(self.sorted_values) - positions_right

            total_pairs = comb(len(self.sorted_values), 2)

        return (  # type: ignore[no-any-return]
            total_pairs - comb(num_strictly_below, 2)
            - comb(num_strictly_above, 2)
        ) / total_pairs


class OutlyingnessBasedDepth(Depth[T]):
    r"""
    Computes depth based on an outlyingness measure.

    An outlyingness function :math:`O(x)` can be converted to a depth
    function as

    .. math::
        D(x) = \frac{1}{1 + O(x)}

    if :math:`O(x)` is unbounded or as

    .. math::
        D(x) = 1 - \frac{O(x)}{\sup O(x)}

    if :math:`O(x)` is bounded. If the infimum value of the
    outlyiness function is not zero, it is subtracted beforehand.

    Args:
        outlyingness (Outlyingness): Outlyingness object.

    References:
        Serfling, R. (2006). Depth functions in nonparametric
        multivariate inference. DIMACS Series in Discrete Mathematics and
        Theoretical Computer Science, 72, 1.

    """

    def __init__(self, outlyingness: Outlyingness[T]):
        self.outlyingness = outlyingness

    def fit(  # noqa: D102
        self,
        X: T,
        y: object = None,
    ) -> OutlyingnessBasedDepth[T]:
        self.outlyingness.fit(X)

        return self

    def transform(self, X: T) -> NDArrayFloat:  # noqa: D102
        outlyingness_values = self.outlyingness.transform(X)

        min_val = self.outlyingness.min
        max_val = self.outlyingness.max

        if math.isinf(max_val):
            return 1 / (1 + outlyingness_values - min_val)

        return 1 - (outlyingness_values - min_val) / (max_val - min_val)


class StahelDonohoOutlyingness(Outlyingness[NDArrayFloat]):
    r"""
    Computes Stahel-Donoho outlyingness.

    Stahel-Donoho outlyingness is defined as

    .. math::
        \sup_{\|u\|=1} \frac{|u^T x - \text{Med}(u^T X))|}{\text{MAD}(u^TX)}

    where :math:`\text{X}` is a sample with distribution :math:`F`,
    :math:`\text{Med}` is the median and :math:`\text{MAD}` is the
    median absolute deviation.

    References:
        Zuo, Y., Cui, H., & He, X. (2004). On the Stahel-Donoho
        estimator and depth-weighted means of multivariate data. Annals of
        Statistics, 32(1), 167–188. https://doi.org/10.1214/aos/1079120132

    """

    def fit(  # noqa: D102
        self,
        X: NDArrayFloat,
        y: object = None,
    ) -> StahelDonohoOutlyingness:

        dim = X.shape[-1]

        if dim == 1:
            self._location = np.median(X, axis=0)
            self._scale = scipy.stats.median_abs_deviation(X, axis=0)
        else:
            raise NotImplementedError("Only implemented for one dimension")

        return self

    def transform(self, X: NDArrayFloat) -> NDArrayFloat:  # noqa: D102

        dim = X.shape[-1]

        if dim == 1:
            # Special case, can be computed exactly
            diff: NDArrayFloat = np.abs(X - self._location) / self._scale

            return diff[..., 0]

        raise NotImplementedError("Only implemented for one dimension")

    @property
    def max(self) -> float:
        return math.inf


class ProjectionDepth(OutlyingnessBasedDepth[NDArrayFloat]):
    """
    Computes Projection depth.

    It is defined as the depth induced by the
    :class:`Stahel-Donoho outlyingness <StahelDonohoOutlyingness>`.

    See also:
        :class:`StahelDonohoOutlyingness`: Stahel-Donoho outlyingness.

    References:
        Zuo, Y., Cui, H., & He, X. (2004). On the Stahel-Donoho
        estimator and depth-weighted means of multivariate data. Annals of
        Statistics, 32(1), 167–188. https://doi.org/10.1214/aos/1079120132

    """

    def __init__(self) -> None:
        super().__init__(outlyingness=StahelDonohoOutlyingness())
