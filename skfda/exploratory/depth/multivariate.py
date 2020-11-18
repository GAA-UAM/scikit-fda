import abc
import math
from scipy.special import comb

import scipy.stats
import sklearn

import numpy as np


class _DepthOrOutlyingness(abc.ABC, sklearn.base.BaseEstimator):
    """
    Abstract class representing a depth or outlyingness function.

    """

    def fit(self, X, y=None):
        """
        Learn the distribution from the observations.

        Args:
            X: Functional dataset from which the distribution of the data is
               inferred.
            y: Unused. Kept only for convention.

        Returns:
            self: Fitted estimator.

        """
        return self

    @abc.abstractmethod
    def predict(self, X):
        """
        Compute the depth or outlyingness inside the learned distribution.

        Args:
            X: Points whose depth is going to be evaluated.

        """
        pass

    def fit_predict(self, X, y=None):
        """
        Compute the depth or outlyingness of each observation with respect to
        the whole dataset.

        Args:
            X: Dataset.
            y: Unused. Kept only for convention.

        """
        return self.fit(X).predict(X)

    def __call__(self, X, *, distribution=None):
        """
        Allows the depth or outlyingness to be used as a function.

        Args:
            X: Points whose depth is going to be evaluated.
            distribution: Functional dataset from which the distribution of
                the data is inferred. If ``None`` it is the same as ``X``.

        """
        copy = sklearn.base.clone(self)

        if distribution is None:
            return copy.fit_predict(X)
        else:
            return copy.fit(distribution).predict(X)

    @property
    def max(self):
        """
        Maximum (or supremum if there is no maximum) of the possibly predicted
        values.

        """
        return 1

    @property
    def min(self):
        """
        Minimum (or infimum if there is no maximum) of the possibly predicted
        values.

        """
        return 0


class Depth(_DepthOrOutlyingness):
    """
    Abstract class representing a depth function.

    """
    pass


class Outlyingness(_DepthOrOutlyingness):
    """
    Abstract class representing an outlyingness function.

    """
    pass


def _searchsorted_one_dim(array, values, *, side='left'):
    searched_index = np.searchsorted(array, values, side=side)

    return searched_index


_searchsorted_vectorized = np.vectorize(
    _searchsorted_one_dim,
    signature='(n),(m),()->(m)',
    excluded='side')


def _searchsorted_ordered(array, values, *, side='left'):
    return _searchsorted_vectorized(array, values, side=side)


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
    return _searchsorted_ordered(np.sort(column), column,
                                 side='right') / len(column)


class _UnivariateFraimanMuniz(Depth):
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

    def fit(self, X, y=None):
        self._sorted_values = np.sort(X, axis=0)
        return self

    def predict(self, X):
        cum_dist = _searchsorted_ordered(
            np.moveaxis(self._sorted_values, 0, -1),
            np.moveaxis(X, 0, -1), side='right') / len(self._sorted_values)

        assert cum_dist.shape[-2] == 1
        return 1 - np.abs(0.5 - np.moveaxis(cum_dist, -1, 0)[..., 0])

    @property
    def min(self):
        return 1 / 2


class SimplicialDepth(Depth):
    r"""
    Simplicial depth.

    The simplicial depth of a point :math:`x` in :math:`\mathbb{R}^p` given a
    distribution :math:`F` is the probability that a random simplex with its
    :math:`p + 1` points sampled from :math:`F` contains :math:`x`.

    References:

        Liu, R. Y. (1990). On a Notion of Data Depth Based on Random
        Simplices. The Annals of Statistics, 18(1), 405–414.


    """

    def fit(self, X, y=None):
        self._dim = X.shape[-1]

        if self._dim == 1:
            self.sorted_values = np.sort(X, axis=0)
        else:
            raise NotImplementedError("SimplicialDepth is currently only "
                                      "implemented for one-dimensional data.")

        return self

    def predict(self, X):

        assert self._dim == X.shape[-1]

        if self._dim == 1:
            positions_left = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1))

            positions_left = np.moveaxis(positions_left, -1, 0)[..., 0]

            positions_right = _searchsorted_ordered(
                np.moveaxis(self.sorted_values, 0, -1),
                np.moveaxis(X, 0, -1), side='right')

            positions_right = np.moveaxis(positions_right, -1, 0)[..., 0]

            num_strictly_below = positions_left
            num_strictly_above = len(self.sorted_values) - positions_right

            total_pairs = comb(len(self.sorted_values), 2)

        return (total_pairs - comb(num_strictly_below, 2)
                - comb(num_strictly_above, 2)) / total_pairs


class OutlyingnessBasedDepth(Depth):
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

    def __init__(self, outlyingness):
        self.outlyingness = outlyingness

    def fit(self, X, y=None):
        self.outlyingness.fit(X)

        return self

    def predict(self, X):
        outlyingness_values = self.outlyingness.predict(X)

        min_val = self.outlyingness.min
        max_val = self.outlyingness.max

        if(math.isinf(max_val)):
            return 1 / (1 + outlyingness_values - min_val)
        else:
            return 1 - (outlyingness_values - min_val) / (max_val - min_val)


class StahelDonohoOutlyingness(Outlyingness):
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

    def fit(self, X, y=None):

        dim = X.shape[-1]

        if dim == 1:
            self._location = np.median(X, axis=0)
            self._scale = scipy.stats.median_abs_deviation(
                X, axis=0)
        else:
            raise NotImplementedError("Only implemented for one dimension")

        return self

    def predict(self, X):

        dim = X.shape[-1]

        if dim == 1:
            # Special case, can be computed exactly
            return (np.abs(X - self._location) /
                    self._scale)[..., 0]

        else:
            raise NotImplementedError("Only implemented for one dimension")

    @property
    def max(self):
        return np.inf


class ProjectionDepth(OutlyingnessBasedDepth):
    r"""
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

    def __init__(self):
        super().__init__(outlyingness=StahelDonohoOutlyingness())
